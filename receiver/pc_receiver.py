#!/usr/bin/env python3
"""
PC-side receiver for the ESP32 fuel logger.

1. Connect this PC to the Wi-Fi network ESP32-FuelLogger (password redacted)
2. Run: python pc_receiver.py (optionally: --host 192.168.4.1 --out logs)

Writes, per ESP32 boot, a folder logs/session_<time>_<bootid>/ containing
  timeseries.csv, raw_can.csv, status.log
Reconnects automatically. A reconnect to the same ESP32 boot keeps appending
to the same folder; an ESP32 reboot starts a new folder.
"""

import argparse
import os
import socket
import time
from datetime import datetime

TIMESERIES_HEADER = (
    "timestamp_ms,latitude,longitude,gps_altitude_m,gps_speed_kmph,gps_valid,"
    "rpm,vehicle_speed_kmph,maf_g_s,fuel_level_pct,fuel_rate_lph,"
    "rpm_valid,speed_valid,maf_valid,fuel_level_valid,fuel_rate_valid"
)
RAW_CAN_HEADER = "timestamp_ms,can_id,dlc,data0,data1,data2,data3,data4,data5,data6,data7"


class Session:
    def __init__(self, out_dir, boot_id):
       stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
       self.path = os.path.join(out_dir, f"session_{stamp}_{boot_id}")
       os.makedirs(self.path, exist_ok=True)
       self.ts = open(os.path.join(self.path, "timeseries.csv"), "w", newline="")
       self.raw = open(os.path.join(self.path, "raw_can.csv"), "w", newline="")
       self.status = open(os.path.join(self.path, "status.log"), "w")
       self.ts.write(TIMESERIES_HEADER + "\n")
       self.raw.write(RAW_CAN_HEADER + "\n")
       self.n_ts = 0
       self.n_raw = 0
       self.last_flush = time.time()
       print(f"[session] writing to {self.path}")

    def handle(self, line):
       if line.startswith("RAW_CAN,"):
           self.raw.write(line[8:] + "\n")
           self.n_raw += 1
       elif line.startswith("TIMESERIES,"):
           self.ts.write(line[11:] + "\n")
           self.n_ts += 1
       elif line.startswith("HEARTBEAT,"):
           self.on_heartbeat(line)
       elif line.startswith("STATUS"):
           self.status.write(f"{datetime.now().isoformat(timespec='seconds')} {line}\n")
           print(f"[esp32] {line}")
       # anything else: ignore

    def on_heartbeat(self, line):
       # HEARTBEAT,seq,uptime_ms,can_driver,can_traffic,can_total,can_last_age_ms,can_fps,gps,ram
       p = line.split(",")
       if len(p) < 10:
           return
       driver, traffic, total, age, fps, gps, ram = p[3], p[4], p[5], p[6], p[7], p[8], p[9]
       self.last_hb = time.time()
       self.hb_count = getattr(self, "hb_count", 0) + 1
       key = (driver, traffic, gps)
       changed = key != getattr(self, "hb_key", None)
       self.hb_key = key
       if changed or self.hb_count % 10 == 0: # on change, else ~every 5 s
           if driver != "RUNNING":
               msg = f"CAN DRIVER PROBLEM: {driver}"
           elif traffic == "ACTIVE":
               msg = f"CAN OK ({fps} frames/s)"
           elif traffic == "NEVER":
               msg = "CAN SILENT: no frames since boot (check wiring/bitrate/ignition)"
           else:
               msg = f"CAN SILENT: last frame {age} ms ago"
           print(f"[hb] link OK | {msg} | total={total} | gps={gps} | esp_buf={ram}")
       self.status.write(f"{datetime.now().isoformat(timespec='seconds')} {line}\n")

    def maybe_flush(self, force=False):
       if force or time.time() - self.last_flush >= 1.0:
           for f in (self.ts, self.raw, self.status):
               f.flush()
           self.last_flush = time.time()

    def close(self):
       self.maybe_flush(force=True)
       for f in (self.ts, self.raw, self.status):
           f.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="192.168.4.1")
    ap.add_argument("--port", type=int, default=9000)
    ap.add_argument("--out", default="logs")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    session = None
    session_boot = None
    last_report = time.time()

    try:
       while True:
           try:
               print(f"[net] connecting to {args.host}:{args.port} ...")
               sock = socket.create_connection((args.host, args.port), timeout=5)
               sock.settimeout(1)
               print("[net] connected")
           except OSError as e:
               print(f"[net] connect failed ({e}); retrying in 2 s")
               time.sleep(2)
               continue

           buf = b""
           last_rx = time.time()
           stall_warned = False
           try:
               while True:
                  try:
                      chunk = sock.recv(8192)
                  except socket.timeout:
                      # ESP32 sends a heartbeat every 0.5 s, so silence = link problem
                      if time.time() - last_rx > 3 and not stall_warned:
                         print("[net] LINK STALL: connected but no data/heartbeat for 3 s (ESP32 hung or out of range?)")
                         stall_warned = True
                      continue
                  if not chunk:
                      print("[net] ESP32 closed the connection")
                      break
                  buf += chunk
                  last_rx = time.time()
                  stall_warned = False
                  *lines, buf = buf.split(b"\n") # last piece = partial line
                  for raw in lines:
                      line = raw.decode("ascii", errors="replace").strip()
                      if not line:
                         continue
                      if line.startswith("ESP32_FUEL_LOGGER"):
                         parts = line.split(",")
                         boot_id = parts[1] if len(parts) > 1 else "unknown"
                         if boot_id != session_boot:
                             if session:
                                 session.close()
                             session = Session(args.out, boot_id)
                             session_boot = boot_id
                         else:
                             print("[session] same ESP32 boot, continuing")
                         continue
                      if session:
                         session.handle(line)
                  if session:
                      session.maybe_flush()
                      if time.time() - last_report >= 5:
                         print(f"[rx] timeseries={session.n_ts} raw_can={session.n_raw}")
                         last_report = time.time()
           except OSError as e:
               print(f"[net] connection error: {e}")
           finally:
               sock.close()
               # a partial trailing line is discarded (ESP32 re-sends unacked batches)
               if session:
                  session.maybe_flush(force=True)
           time.sleep(1)
    except KeyboardInterrupt:
       print("\n[exit] stopping")
    finally:
       if session:
           session.close()


if __name__ == "__main__":
    main()
