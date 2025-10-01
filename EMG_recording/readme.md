# EMG recording from the custom emg sensor:
The sensor currently sends 6 raw channel streams over serial port (115200 BAUD).

# Windows
We use https://hackaday.io/project/5334-serialplot-realtime-plotting-software, the serialplot software
In the record tab, make sure the timestamp is enabled and the format and output location are set correctly:

<img width="607" height="245" alt="image" src="https://github.com/user-attachments/assets/aa802243-4bde-475e-b61c-a48c7f642c9e" />

Start recording with the red button


# MAC

Download CoolTerm app for Mac at https://coolterm.macupdate.com

<img width="1680" height="1050" alt="Screenshot 2025-09-24 at 19 46 31" src="https://github.com/user-attachments/assets/a90d1e1f-51c1-488a-823d-b1da28a8a561" />

<img width="1680" height="1050" alt="Screenshot 2025-09-24 at 19 46 23" src="https://github.com/user-attachments/assets/122f70b9-2633-4e75-988a-d04be30511e3" />

Once these settings are set up, select the right usb port and just press the "connect" button on the top left to start streaming the data.
To save the data go to Connection/File Capture/Start as shown below. Select the location where to save it.

<img width="1680" height="1050" alt="Screenshot 2025-09-24 at 20 03 23" src="https://github.com/user-attachments/assets/fbe8904c-e491-4c73-a589-3f7d61aaad90" />

Since it is saving a txt file (instead of a CSV), with a slighty different timestamp format, you could think of convert it to CSV with UNIX timestamp format using a simple python code.

