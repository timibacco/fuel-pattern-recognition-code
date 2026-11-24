# Fuel Consumption Pattern Recognition — Source Code

Source code for the B.Sc. thesis *Development of a Recognition Model for Vehicle Fuel Consumption Pattern in a Controlled Spatiotemporal Environment* (Abayomi, O. S., Department of Computer Engineering, Obafemi Awolowo University, Ile-Ife, 2026).

| Folder | Thesis appendix | Contents |
|---|---|---|
| [`firmware/`](firmware/) | Appendix A | ESP32 logger firmware (Arduino framework, 38-pin ESP32 board): OBD-II polling over CAN (TWAI + SN65HVD230), NEO-6M GPS parsing, 100 ms time-series records, RAM ring buffer and Wi-Fi/TCP streaming. |
| [`receiver/`](receiver/) | Appendix B | Python program that connects to the logger's Wi-Fi access point, receives the TCP stream and writes one session folder (`timeseries.csv`, `raw_can.csv`, `status.log`) per logger boot. |
| [`evaluation/`](evaluation/) | Appendix C | Google Colab cell that builds the purged temporal split, pretrains the LSTM autoencoder, trains the segment classifier over five seeds and evaluates it against a majority-class predictor and a random forest. |

## Hardware
ESP32-WROOM-32 (38-pin development board) · SN65HVD230 CAN transceiver · NEO-6M GPS receiver · LM2596 buck converter (12 V → 5 V) · OBD-II pigtail cable. CAN bus at 500 kbit/s.

## Running
**Firmware** — open `firmware/firmware.ino` in the Arduino IDE with the ESP32 board package installed, set `AP_PASSWORD`, and upload.

**Receiver** — join the `ESP32-FuelLogger` Wi-Fi network, then:
```
python receiver/pc_receiver.py --host 192.168.4.1 --out logs
```

**Evaluation** — run `evaluation/honest_eval_cell.py` in Google Colab (PyTorch, scikit-learn, pandas, NumPy). Set `CSV_PATH` to the merged feature table from the data repository.

## Data
The recorded sessions and landmark coordinates are in the companion repository: https://github.com/timibacco/fuel-pattern-recognition-data

## Licence
MIT — see [LICENSE](LICENSE).
