/*
  ESP32 VEHICLE FUEL CONSUMPTION DATA LOGGER (WiFi-only, CAN listen-only)

  Hardware
  --------
  ESP32 38-pin classic ESP32 / WROOM

  GPS NEO-6M -> UART2 (pins set by GPS_RX_PIN / GPS_TX_PIN below)
  SN65HVD230 CAN xcvr -> TWAI (pins set by CAN_TX_PIN / CAN_RX_PIN below)
    CANH -> OBD-II pin 6
    CANL -> OBD-II pin 14

  Architecture
  ------------
  1. NO SD card. Nothing is stored on the ESP32.
  2. ESP32 runs its own Wi-Fi access point and a TCP server (port 9000).
  3. The PC connects (pc_receiver.py) and saves timeseries.csv and
     raw_can.csv.
  4. Every record goes into a RAM ring buffer first; the buffer is drained
     to the PC in batches. If the PC is not connected (or is too slow) the
     buffer holds records; when full, the OLDEST are dropped and reported.
  5. CAN is LISTEN-ONLY: this device never transmits on the vehicle bus.

  Wire protocol (one text line per record, '\n' terminated)
  ---------------------------------------------------------
    ESP32_FUEL_LOGGER,<boot_id_hex> first line after connect
    STATUS,... periodic diagnostics
    RAW_CAN,<ts_ms>,<id_hex>,<dlc>,d0..d7
    TIMESERIES,<ts_ms>,lat,lon,alt,gps_kmph,gps_valid,rpm,speed,maf,
              rpm_valid,speed_valid,maf_valid

  NOTE: because we never transmit, the vehicle will not answer with
  OBD-II PID responses. rpm / speed / maf will therefore be INVALID
  (flag = 0) until you fill in decodeProprietaryIfConfigured() with
  verified IDs for your vehicle. Everything on the bus is still captured
  in raw_can.csv. GPS speed is available regardless.
*/

#include <Arduino.h>
#include <WiFi.h>
#include <HardwareSerial.h>
#include "driver/twai.h"
#include "esp_system.h"

// ============================================================
// PIN CONFIGURATION (unchanged from previous version)
// ============================================================

static const int GPS_RX_PIN = 17; // ESP32 pin that receives GPS TX
static const int GPS_TX_PIN = 16;

static const int CAN_TX_PIN = 25; // -> transceiver CTX (swapped 2026-09-30 to match physical wiring)
static const int CAN_RX_PIN = 26; // -> transceiver CRX

// ============================================================
// TIMING
// ============================================================

static const uint32_t SAMPLE_INTERVAL_MS = 100; // 10 Hz timeseries
static const uint32_t STATUS_INTERVAL_MS = 5000;
static const uint32_t HEARTBEAT_INTERVAL_MS = 500; // sent always: connect, idle, CAN dead or alive
static const uint32_t CAN_ACTIVE_WINDOW_MS = 1000; // CAN counted ACTIVE if a frame arrived within this
static const uint32_t DRAIN_INTERVAL_MS = 20; // batch TCP sends
static const uint32_t STALE_MS = 2000; // invalidate old data

// ============================================================
// WIFI
// ============================================================

const char *AP_SSID = "ESP32-FuelLogger";
const char *AP_PASSWORD = "********"; // redacted for the thesis
static const uint16_t TCP_PORT = 9000;

WiFiServer wifiServer(TCP_PORT);
WiFiClient wifiClient;

uint32_t bootId = 0;

// ============================================================
// GPS
// ============================================================

HardwareSerial GPS(2);

struct GPSState {
  bool valid = false;
  double latitude = 0.0;
  double longitude = 0.0;
  float altitude = 0.0f;
  float speedKmph = 0.0f;
  uint32_t lastFixMs = 0;
};

GPSState gpsState;

// Diagnostics: proves whether raw bytes are arriving at all, independent
// of whether they ever produce a valid fix. Remove once GPS is confirmed.
uint32_t gpsBytesReceived = 0;
char gpsLastLine[96] = "";

// ============================================================
// VEHICLE STATE
// ============================================================

struct VehicleState {
  bool rpmValid = false;
  float rpm = 0.0f;
  bool speedValid = false;
  float vehicleSpeedKmph = 0.0f;
  bool mafValid = false;
  float mafGramsPerSec = 0.0f;
  bool fuelLevelValid = false;
  float fuelLevelPercent = 0.0f;
  bool fuelRateValid = false;
  float fuelRateLph = 0.0f;
  uint32_t lastUpdateMs = 0;
};

VehicleState vehicleState;

// ============================================================
// RAM RING BUFFER
// ============================================================
//
// All records pass through here. 384 x 128 B = 48 KB.
// Longest record (TIMESERIES) is ~100 chars, so 128 is safe.
//

static const size_t RAM_RECORD_SIZE = 160;
static const size_t RAM_BUFFER_CAPACITY = 384;

struct BufferedRecord {
  char data[RAM_RECORD_SIZE];
};

BufferedRecord ramBuffer[RAM_BUFFER_CAPACITY];

size_t ramHead = 0;
size_t ramCount = 0;

uint32_t ramDroppedRecords = 0;
uint32_t lastDropReportMs = 0;
uint32_t droppedAtLastReport = 0;

// ============================================================
// STATISTICS
// ============================================================

uint32_t totalCanFrames = 0;
uint32_t totalTimeseriesSamples = 0;
uint32_t wifiRecordsSent = 0;
uint32_t lastStatus = 0;
uint32_t lastDrain = 0;

// Heartbeat / CAN liveness
uint32_t heartbeatSeq = 0;
uint32_t lastHeartbeatMs = 0;
uint32_t lastCanFrameMs = 0; // 0 = no frame seen since boot
uint32_t framesAtLastHeartbeat = 0;
bool canInitOk = false;

static char txBuf[1400]; // one TCP segment worth

// ============================================================
// LOGGING
// ============================================================

void logInfo(const char *category, const char *message) {
  Serial.print("[");
  Serial.print(category);
  Serial.print("] ");
  Serial.println(message);
}

void logError(const char *category, const char *message) {
  Serial.print("[ERROR][");
  Serial.print(category);
  Serial.print("] ");
  Serial.println(message);
}

// ============================================================
// RAM BUFFER OPERATIONS
// ============================================================

void bufferRecord(const char *record) {

  if (ramCount >= RAM_BUFFER_CAPACITY) {

    // Explicitly discard the oldest record.
    ramHead = (ramHead + 1) % RAM_BUFFER_CAPACITY;
    ramCount--;
    ramDroppedRecords++;

    // Rate-limited so a busy CAN bus can't flood the serial port.
    uint32_t now = millis();
    if (now - lastDropReportMs >= 1000) {
     Serial.print("[ERROR][BUFFER] RAM BUFFER FULL. Oldest records discarded. Total dropped: ");
     Serial.println(ramDroppedRecords);
     lastDropReportMs = now;
    }
  }

  size_t tail = (ramHead + ramCount) % RAM_BUFFER_CAPACITY;

  strncpy(ramBuffer[tail].data, record, RAM_RECORD_SIZE - 1);
  ramBuffer[tail].data[RAM_RECORD_SIZE - 1] = '\0';

  ramCount++;
}

// ============================================================
// WIFI
// ============================================================

// ============================================================
// HEARTBEAT
// ============================================================
//
// Sent every HEARTBEAT_INTERVAL_MS regardless of CAN/GPS state, so the PC
// can tell "link alive, CAN silent" from "link dead". Fields:
// HEARTBEAT,seq,uptime_ms,can_driver,can_traffic,can_total,can_last_age_ms,can_fps,gps,ram
// can_driver : INIT_FAILED | RUNNING | STOPPED | BUS_OFF | RECOVERING
// can_traffic: ACTIVE (frame within last 1 s) | SILENT (none) | NEVER (none since boot)
//
void sendHeartbeat(bool force = false) {

  uint32_t now = millis();

  if (!force && now - lastHeartbeatMs < HEARTBEAT_INTERVAL_MS) {
    return;
  }

  uint32_t elapsed = now - lastHeartbeatMs;
  lastHeartbeatMs = now;

  const char *driver = "INIT_FAILED";
  if (canInitOk) {
    twai_status_info_t info = {};
    twai_get_status_info(&info);
    switch (info.state) {
     case TWAI_STATE_RUNNING: driver = "RUNNING"; break;
     case TWAI_STATE_BUS_OFF: driver = "BUS_OFF"; break;
     case TWAI_STATE_RECOVERING: driver = "RECOVERING"; break;
     default: driver = "STOPPED"; break;
    }
  }

  uint32_t age = lastCanFrameMs == 0 ? 0 : now - lastCanFrameMs;
  const char *traffic =
    lastCanFrameMs == 0 ? "NEVER" :
    (age <= CAN_ACTIVE_WINDOW_MS ? "ACTIVE" : "SILENT");

  uint32_t fps = 0;
  if (elapsed > 0 && elapsed < 10000) {
    fps = (uint32_t)((uint64_t)(totalCanFrames - framesAtLastHeartbeat) * 1000ULL / elapsed);
  }
  framesAtLastHeartbeat = totalCanFrames;

  char rec[RAM_RECORD_SIZE];

  snprintf(
    rec, sizeof(rec),
    "HEARTBEAT,%lu,%lu,%s,%s,%lu,%lu,%lu,%s,%u",
    (unsigned long)heartbeatSeq++,
    (unsigned long)now,
    driver,
    traffic,
    (unsigned long)totalCanFrames,
    (unsigned long)age,
    (unsigned long)fps,
    gpsState.valid ? "VALID" : "NO_FIX",
    (unsigned)ramCount
  );

  bufferRecord(rec);
}

bool clientConnected() {
  return wifiClient && wifiClient.connected();
}

void startWiFiAP() {

  WiFi.mode(WIFI_AP);

  if (!WiFi.softAP(AP_SSID, AP_PASSWORD)) {
    logError("WIFI", "Failed to start access point.");
    return;
  }

  wifiServer.begin();

  Serial.println("[WIFI] Access point started.");
  Serial.print("[WIFI] SSID: ");
  Serial.println(AP_SSID);
  Serial.print("[WIFI] IP: ");
  Serial.println(WiFi.softAPIP());
  Serial.print("[WIFI] TCP port: ");
  Serial.println(TCP_PORT);
}

void checkWiFiClient() {

  if (wifiClient && !wifiClient.connected()) {
    wifiClient.stop();
    logInfo("WIFI", "Laptop disconnected. Buffering in RAM.");
  }

  if (!clientConnected()) {

    WiFiClient newClient = wifiServer.available();

    if (newClient) {

     wifiClient = newClient;
     wifiClient.setNoDelay(true);

     logInfo("WIFI", "Laptop connected.");
     Serial.print("[WIFI] Client IP: ");
     Serial.println(wifiClient.remoteIP());

     // Handshake. boot_id lets the PC tell a reconnect (same boot,
     // continuing timestamps) from an ESP32 reboot (timestamps restart).
     wifiClient.printf("ESP32_FUEL_LOGGER,%08lX\n", (unsigned long)bootId);
     wifiClient.println("STATUS,WIFI_STREAM_READY");

     // First heartbeat immediately so the PC sees liveness on connect.
     sendHeartbeat(true);
    }
  }
}

// Sends buffered records to the PC in batches. A record is only removed
// from the ring buffer after the whole batch was accepted by TCP.
void drainToWiFi() {

  if (!clientConnected() || ramCount == 0) {
    return;
  }

  uint32_t now = millis();
  if (now - lastDrain < DRAIN_INTERVAL_MS) {
    return;
  }
  lastDrain = now;

  for (int batch = 0; batch < 4 && ramCount > 0; batch++) {

    size_t len = 0;
    size_t n = 0;

    while (n < ramCount) {

     const char *rec = ramBuffer[(ramHead + n) % RAM_BUFFER_CAPACITY].data;
     size_t l = strlen(rec);

     if (len + l + 1 > sizeof(txBuf)) {
       break;
     }

     memcpy(txBuf + len, rec, l);
     len += l;
     txBuf[len++] = '\n';
     n++;
    }

    if (n == 0) {
     return;
    }

    size_t written = wifiClient.write((const uint8_t *)txBuf, len);

    if (written != len) {
     // Records stay in the buffer and are re-sent to the next client.
     logError("WIFI", "TCP write failed. Dropping client.");
     wifiClient.stop();
     return;
    }

    ramHead = (ramHead + n) % RAM_BUFFER_CAPACITY;
    ramCount -= n;
    wifiRecordsSent += n;
  }
}

// ============================================================
// CAN
// ============================================================

// Only reachable if some OTHER tool on the bus polls OBD-II and the
// ECU answers; we never send requests ourselves. Harmless otherwise.
// ------------------------------------------------------------
// ACTIVE OBD-II POLLING (Mode 01, round-robin, non-blocking)
// ------------------------------------------------------------
// Add/remove PIDs here. Response is NOT waited for here -- it arrives
// asynchronously and is picked up by decodeOBDPID() below, called from
// processCAN() every loop pass.
//
// Common PIDs you may want for a fuel-consumption model beyond rpm/
// speed/maf (add to the array and extend decodeOBDPID() to match):
// 0x04 Calculated engine load (A*100/255 %)
// 0x05 Coolant temperature (A-40 degC)
// 0x06/0x07 Short/long term fuel trim bank 1 ((A-128)*100/128 %)
// 0x0F Intake air temperature (A-40 degC)
// 0x11 Throttle position (A*100/255 %)
// 0x2F Fuel tank level input (A*100/255 %)
// 0x5E Engine fuel rate (((A*256)+B)/20 L/h) -- if supported,
// this is a DIRECT fuel-rate signal and is worth checking for
// before deriving fuel rate from MAF (see LSTM note below).
static const uint8_t OBD_POLL_PIDS[] = { 0x0C, 0x0D, 0x10, 0x2F, 0x5E };
// RPM, Speed, MAF, Fuel Tank Level %, Engine Fuel Rate L/h (standard PIDs --
// 0x2F/0x5E are SAE J1979, same 0x7DF/0x7E8 mechanism as the others, not
// guaranteed to be answered on this ECU. 0x5E is a DIRECT fuel-rate signal
// if supported -- prefer it over any MAF-derived estimate as an LSTM label.
static const uint8_t OBD_POLL_PID_COUNT = sizeof(OBD_POLL_PIDS) / sizeof(OBD_POLL_PIDS[0]);
static const uint32_t OBD_POLL_INTERVAL_MS = 100; // one PID request per tick, round-robin

uint8_t obdPollIndex = 0;
uint32_t lastOBDPollMs = 0;

void pollOBDIfDue() {

  uint32_t now = millis();

  if (now - lastOBDPollMs < OBD_POLL_INTERVAL_MS) {
    return;
  }

  lastOBDPollMs = now;

  uint8_t pid = OBD_POLL_PIDS[obdPollIndex];
  obdPollIndex = (obdPollIndex + 1) % OBD_POLL_PID_COUNT;

  twai_message_t tx_msg = {};
  tx_msg.identifier = 0x7DF; // functional broadcast request
  tx_msg.data_length_code = 8;
  tx_msg.data[0] = 0x02; // additional bytes
  tx_msg.data[1] = 0x01; // Mode 01: show current data
  tx_msg.data[2] = pid;
  // data[3..7] left at 0 (padding)

  if (twai_transmit(&tx_msg, pdMS_TO_TICKS(10)) != ESP_OK) {
    // Non-fatal -- just try again on the next scheduled poll.
    logError("CAN", "OBD PID request transmit failed.");
  }
}

void decodeOBDPID(uint32_t id, uint8_t dlc, const uint8_t *data) {

  if (id < 0x7E8 || id > 0x7EF) return;
  if (dlc < 4) return;
  if (data[1] != 0x41) return;

  uint8_t pid = data[2];

  if (pid == 0x0C && dlc >= 5) {
    vehicleState.rpm = (((uint16_t)data[3] << 8) | data[4]) / 4.0f;
    vehicleState.rpmValid = true;
    vehicleState.lastUpdateMs = millis();
  }
  else if (pid == 0x0D) {
    vehicleState.vehicleSpeedKmph = data[3];
    vehicleState.speedValid = true;
    vehicleState.lastUpdateMs = millis();
  }
  else if (pid == 0x10 && dlc >= 5) {
    vehicleState.mafGramsPerSec = (((uint16_t)data[3] << 8) | data[4]) / 100.0f;
    vehicleState.mafValid = true;
    vehicleState.lastUpdateMs = millis();
  }
  else if (pid == 0x2F) {
    // Fuel Tank Level Input: A * 100 / 255 (%)
    vehicleState.fuelLevelPercent = data[3] * 100.0f / 255.0f;
    vehicleState.fuelLevelValid = true;
    vehicleState.lastUpdateMs = millis();
  }
  else if (pid == 0x5E && dlc >= 5) {
    // Engine Fuel Rate: ((A*256)+B) / 20 (L/h) -- direct signal, not MAF-derived.
    vehicleState.fuelRateLph = (((uint16_t)data[3] << 8) | data[4]) / 20.0f;
    vehicleState.fuelRateValid = true;
    vehicleState.lastUpdateMs = millis();
  }
}

// ------------------------------------------------------------
// PLACEHOLDER: proprietary (passive) decode for your Toyota model.
// Fill in only with IDs/offsets/scales you verified against a known
// reference (e.g. correlate raw_can.csv against a scan tool or GPS
// speed using cabana / opendbc). Example skeleton:
//
// if (id == 0x2C4 && dlc >= 8) {
// uint16_t raw = ((uint16_t)data[0] << 8) | data[1];
// vehicleState.rpm = raw * SCALE + OFFSET;
// vehicleState.rpmValid = true;
// vehicleState.lastUpdateMs = millis();
// }
// ------------------------------------------------------------
void decodeProprietaryIfConfigured(uint32_t id, uint8_t dlc, const uint8_t *data) {
  (void)id; (void)dlc; (void)data;
}

void invalidateStaleReadings() {

  uint32_t now = millis();

  if (vehicleState.lastUpdateMs != 0 &&
     now - vehicleState.lastUpdateMs > STALE_MS) {
    vehicleState.rpmValid = false;
    vehicleState.speedValid = false;
    vehicleState.mafValid = false;
  }

  if (gpsState.valid && now - gpsState.lastFixMs > STALE_MS) {
    gpsState.valid = false;
  }
}

// ------------------------------------------------------------
// CAN ID ALLOW-LIST
// ------------------------------------------------------------
// From the RX350 reverse-engineering report, Table 2: core + optional +
// candidate IDs worth keeping for the fuel model. Of 111 IDs seen in the
// pilot, 54 were static (no information) and 76 unmapped -- this cuts
// raw_can.csv down to the ones that matter instead of logging everything.
// Set CAN_FILTER_ENABLED to false to go back to logging every ID.
static const bool CAN_FILTER_ENABLED = true;

static const uint32_t CAN_ID_ALLOWLIST[] = {
  0x1C4, // ENGINE_RPM (core)
  0x2C1, // GAS_PEDAL (core)
  0x2C6, // Pedal flag (core)
  0x0B4, // SPEED (core)
  0x0AA, // WHEEL_SPEEDS (core -- rear-right field is a known sentinel, see report 5)
  0x224, // Brake (core)
  0x3BC, // GEAR_PACKET (core)
  0x024, // KINEMATICS (core)
  0x025, // STEER_ANGLE_SENSOR (core)
  0x260, // STEER_TORQUE_SENSOR (optional)
  0x3B0, // Ambient temp (optional)
  0x381, // HVAC state (optional)
  0x611, // UI_SETTING / odometer (optional)
  0x361, // Unresolved, pedal-like, r=0.99 with pedal (candidate)
  0x1E2, // Unresolved, pedal-like, r=0.93 with pedal (candidate)
  0x3B3, // RPM copy + unknown (candidate)
  0x1D0, // Unresolved, gear/state dependent (candidate)
};
static const uint8_t CAN_ID_ALLOWLIST_COUNT =
  sizeof(CAN_ID_ALLOWLIST) / sizeof(CAN_ID_ALLOWLIST[0]);

bool idIsAllowed(uint32_t id) {
  if (!CAN_FILTER_ENABLED) return true;
  // DEBUG 2026-10-01: temporarily let 0x7E8/0x7E9 (OBD-II Mode 01 response
  // range) through so raw_can.csv shows the actual bytes behind rpm/maf
  // reading 0 despite valid=1 while the engine was confirmed running.
  // Remove once that's root-caused.
  if (id == 0x7E8 || id == 0x7E9) return true;
  for (uint8_t i = 0; i < CAN_ID_ALLOWLIST_COUNT; i++) {
    if (CAN_ID_ALLOWLIST[i] == id) return true;
  }
  return false;
}

void processCAN() {

  twai_message_t message;

  // Bounded so a very busy bus can't starve the rest of loop().
  for (int i = 0; i < 200 && twai_receive(&message, 0) == ESP_OK; i++) {

    // Ignore remote-request and error-type frames' payload (no data).
    totalCanFrames++;
    lastCanFrameMs = millis();

    uint32_t timestamp = millis();
    uint32_t id = message.identifier;
    uint8_t dlc = message.data_length_code;
    if (dlc > 8) dlc = 8;

    // decodeOBDPID runs on every frame regardless of the allow-list --
    // it only triggers on the 0x7E8-0x7EF OBD response range, which the
    // list above isn't meant to filter.
    decodeOBDPID(id, dlc, message.data);

    if (idIsAllowed(id)) {

     char record[RAM_RECORD_SIZE];

     snprintf(
       record, sizeof(record),
       "RAW_CAN,%lu,%lX,%u,%u,%u,%u,%u,%u,%u,%u,%u",
       (unsigned long)timestamp,
       (unsigned long)id,
       dlc,
       dlc > 0 ? message.data[0] : 0,
       dlc > 1 ? message.data[1] : 0,
       dlc > 2 ? message.data[2] : 0,
       dlc > 3 ? message.data[3] : 0,
       dlc > 4 ? message.data[4] : 0,
       dlc > 5 ? message.data[5] : 0,
       dlc > 6 ? message.data[6] : 0,
       dlc > 7 ? message.data[7] : 0
     );

     bufferRecord(record);
    }
    decodeProprietaryIfConfigured(id, dlc, message.data);
  }
}

bool initializeCAN() {

  // NORMAL: transmits Mode 01 PID requests and ACKs responses, while still
  // passively receiving every other frame on the bus exactly as before --
  // CAN is a broadcast bus, so NORMAL mode loses nothing from raw_can.csv.
  twai_general_config_t generalConfig =
    TWAI_GENERAL_CONFIG_DEFAULT(
     (gpio_num_t)CAN_TX_PIN,
     (gpio_num_t)CAN_RX_PIN,
     TWAI_MODE_NORMAL
    );

  // Default RX queue is only 5 frames -- far too small for a live
  // vehicle bus. Enlarge so bursts survive between loop() passes.
  generalConfig.rx_queue_len = 256;

  twai_timing_config_t timingConfig = TWAI_TIMING_CONFIG_500KBITS();
  twai_filter_config_t filterConfig = TWAI_FILTER_CONFIG_ACCEPT_ALL();

  if (twai_driver_install(&generalConfig, &timingConfig, &filterConfig) != ESP_OK) {
    logError("CAN", "TWAI driver installation failed.");
    return false;
  }

  if (twai_start() != ESP_OK) {
    logError("CAN", "TWAI start failed.");
    twai_driver_uninstall();
    return false;
  }

  logInfo("CAN", "Listen-only CAN started at 500 kbit/s.");
  canInitOk = true;
  return true;
}

// ============================================================
// GPS
// ============================================================

double nmeaToDecimal(const char *value, char direction) {

  if (!value || strlen(value) < 3) {
    return 0.0;
  }

  double raw = atof(value);
  int degrees = (int)(raw / 100.0);
  double minutes = raw - (degrees * 100.0);
  double decimal = degrees + minutes / 60.0;

  if (direction == 'S' || direction == 'W') {
    decimal = -decimal;
  }

  return decimal;
}

// Splits on commas KEEPING empty fields (strtok would collapse them
// and shift every later field index). Modifies 'line' in place.
int splitFields(char *line, char **fields, int maxFields) {

  int count = 0;
  char *p = line;

  while (count < maxFields) {

    fields[count++] = p;

    char *comma = strchr(p, ',');

    if (!comma) {
     break;
    }

    *comma = '\0';
    p = comma + 1;
  }

  return count;
}

void processGPSLine(char *line) {

  bool isRMC = strncmp(line + 3, "RMC", 3) == 0;
  bool isGGA = strncmp(line + 3, "GGA", 3) == 0;

  if (line[0] != '$' || (!isRMC && !isGGA)) {
    return;
  }

  char *fields[20];
  int count = splitFields(line, fields, 20);

  if (isRMC) {
    // 2 status, 3 lat, 4 N/S, 5 lon, 6 E/W, 7 speed knots
    if (count >= 8) {
     if (fields[2][0] == 'A') {
       gpsState.latitude = nmeaToDecimal(fields[3], fields[4][0]);
       gpsState.longitude = nmeaToDecimal(fields[5], fields[6][0]);
       gpsState.speedKmph = atof(fields[7]) * 1.852f;
       gpsState.valid = true;
       gpsState.lastFixMs = millis();
     } else {
       gpsState.valid = false;
     }
    }
  }
  else {
    // 6 fix quality, 9 altitude
    if (count >= 10 && atoi(fields[6]) > 0) {
     gpsState.altitude = atof(fields[9]);
    }
  }
}

void processGPS() {

  static char line[160];
  static size_t index = 0;

  while (GPS.available()) {

    char c = GPS.read();
    gpsBytesReceived++;

    if (c == '\n') {
     line[index] = '\0';
     if (index > 6) {
       strncpy(gpsLastLine, line, sizeof(gpsLastLine) - 1);
       gpsLastLine[sizeof(gpsLastLine) - 1] = '\0';
       processGPSLine(line);
     }
     index = 0;
    }
    else if (c != '\r') {
     if (index < sizeof(line) - 1) {
       line[index++] = c;
     } else {
       index = 0; // malformed line, discard
     }
    }
  }
}

// ============================================================
// TIMESERIES SAMPLE
// ============================================================

void generateTimeseriesSample() {

  char record[RAM_RECORD_SIZE];

  snprintf(
    record, sizeof(record),
    "TIMESERIES,%lu,%.7f,%.7f,%.2f,%.2f,%d,%.2f,%.2f,%.4f,%.2f,%.2f,%d,%d,%d,%d,%d",
    (unsigned long)millis(),
    gpsState.latitude,
    gpsState.longitude,
    gpsState.altitude,
    gpsState.speedKmph,
    gpsState.valid ? 1 : 0,
    vehicleState.rpm,
    vehicleState.vehicleSpeedKmph,
    vehicleState.mafGramsPerSec,
    vehicleState.fuelLevelPercent,
    vehicleState.fuelRateLph,
    vehicleState.rpmValid ? 1 : 0,
    vehicleState.speedValid ? 1 : 0,
    vehicleState.mafValid ? 1 : 0,
    vehicleState.fuelLevelValid ? 1 : 0,
    vehicleState.fuelRateValid ? 1 : 0
  );

  bufferRecord(record);
  totalTimeseriesSamples++;
}

// ============================================================
// STATUS
// ============================================================

void printStatus() {

  uint32_t now = millis();

  if (now - lastStatus < STATUS_INTERVAL_MS) {
    return;
  }

  lastStatus = now;

  twai_status_info_t can = {};
  twai_get_status_info(&can);

  Serial.println();
  Serial.println("========== SYSTEM STATUS ==========");
  Serial.printf("Uptime: %lu s\n", (unsigned long)(now / 1000));
  Serial.printf("CAN frames: %lu (rx missed by driver: %lu)\n",
               (unsigned long)totalCanFrames, (unsigned long)can.rx_missed_count);
  Serial.printf("Timeseries samples: %lu\n", (unsigned long)totalTimeseriesSamples);
  Serial.printf("Wi-Fi client: %s\n", clientConnected() ? "CONNECTED" : "NOT CONNECTED");
  Serial.printf("RAM buffer: %u/%u\n", (unsigned)ramCount, (unsigned)RAM_BUFFER_CAPACITY);
  Serial.printf("RAM dropped: %lu\n", (unsigned long)ramDroppedRecords);
  Serial.printf("Wi-Fi records sent: %lu\n", (unsigned long)wifiRecordsSent);
  Serial.printf("GPS: %s (raw bytes rx: %lu)\n", gpsState.valid ? "VALID" : "NO FIX",
               (unsigned long)gpsBytesReceived);
  Serial.printf("GPS last sentence: %s\n", gpsLastLine[0] ? gpsLastLine : "(none yet)");
  Serial.println("===================================");
  Serial.println();

  // Status goes through the same ordered buffer as data.
  char statusRecord[RAM_RECORD_SIZE];

  snprintf(
    statusRecord, sizeof(statusRecord),
    "STATUS,uptime_s=%lu,gps=%s,can_frames=%lu,can_missed=%lu,dropped=%lu,ram=%u",
    (unsigned long)(now / 1000),
    gpsState.valid ? "VALID" : "NO_FIX",
    (unsigned long)totalCanFrames,
    (unsigned long)can.rx_missed_count,
    (unsigned long)ramDroppedRecords,
    (unsigned)ramCount
  );

  bufferRecord(statusRecord);
}

// ============================================================
// SETUP / LOOP
// ============================================================

static uint32_t lastSample = 0;

void setup() {

  Serial.begin(115200);
  delay(1000);

  Serial.println();
  Serial.println("==========================================");
  Serial.println(" ESP32 VEHICLE FUEL LOGGER (WiFi, listen-only)");
  Serial.println("==========================================");

  bootId = esp_random();

  GPS.begin(9600, SERIAL_8N1, GPS_RX_PIN, GPS_TX_PIN);
  logInfo("GPS", "GPS serial initialized.");

  if (!initializeCAN()) {
    logError("CAN", "CAN initialization failed.");
  }

  startWiFiAP();

  logInfo("WIFI", "Waiting for laptop TCP client on port 9000.");

  lastSample = millis();

  logInfo("BOOT", "System initialization complete.");
}

void loop() {

  processGPS();
  processCAN();
  pollOBDIfDue();
  invalidateStaleReadings();
  checkWiFiClient();

  uint32_t now = millis();

  if (now - lastSample >= SAMPLE_INTERVAL_MS) {

    // If we fell far behind, resync instead of emitting a burst.
    if (now - lastSample > 5 * SAMPLE_INTERVAL_MS) {
     lastSample = now;
    } else {
     lastSample += SAMPLE_INTERVAL_MS;
    }

    generateTimeseriesSample();
  }

  sendHeartbeat();
  drainToWiFi();
  printStatus();

  delay(1);
}
