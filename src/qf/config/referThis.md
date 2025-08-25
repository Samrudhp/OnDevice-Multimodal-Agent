# referThis.md

Purpose
-------
Quick reference for getting a mobile device to reach the QuadFusion backend running on a different PC. Put this on the development PC that will run the backend.

Prerequisites
-------------
- Backend code available and uvicorn installed (Python env where server runs).
- Phone and PC on the same Wi‑Fi / LAN (or use ngrok if not).
- Allow opening firewall port 8000 temporarily for testing.

Steps
-----
1) Start the backend so it listens on all interfaces

```powershell
# from repo src/backend
uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
```

2) Find your PC LAN IP (IPv4)

```powershell
ipconfig
# Look for "IPv4 Address" for the active adapter (example: 192.168.1.42)
```

3) (Windows) Allow port 8000 through firewall (temporary for testing)

```powershell
# Run as Administrator
New-NetFirewallRule -DisplayName "AllowQuadFusion8000" -Direction Inbound -LocalPort 8000 -Protocol TCP -Action Allow
```

4) Verify backend works locally on the PC

```powershell
# health check
Invoke-RestMethod -Uri http://localhost:8000/health -UseBasicParsing

# test a valid realtime request
$body = '{"session_id":"test-session","timestamp":"2025-01-01T00:00:00Z","sensor_data":{"touch_events":[],"keystroke_events":[],"app_usage":[]}}'
Invoke-WebRequest -Uri http://localhost:8000/api/v1/process/realtime -Method POST -ContentType 'application/json' -Body $body -UseBasicParsing | Select-Object -ExpandProperty Content
```

5) Test from your phone's browser

- On the phone open: `http://<PC_IP>:8000/docs` (replace `<PC_IP>` with the IPv4 from step 2).
- If the Swagger UI loads, the network and firewall are OK.

6) Configure the mobile app to use the PC IP

Options:
- Quick runtime override (recommended for testing) — add early in app init (App.tsx):

```ts
import { setApiBaseUrl } from './src/qf/config/api';
setApiBaseUrl('http://192.168.X.Y:8000');
```

- Or use the helper we added:

```ts
import { setDeviceIp } from './src/qf/config/api';
setDeviceIp('192.168.X.Y:8000');
```

- Or edit `src/qf/config/api.ts` and set `API_CONFIG.DEVICE` to your PC IP.

7) Expo specific

- In Expo DevTools choose "LAN" (not Tunnel) so the packager uses the LAN IP.
- Our code attempts to infer the debuggerHost when `expo-constants` is available.

8) If direct LAN access is not possible (different networks, NAT or strict routing)

Use ngrok as a quick workaround:

```powershell
# run on the backend PC
ngrok http 8000
# ngrok returns a public https URL like https://abcd-1234.ngrok.io
# then in app: setApiBaseUrl('https://abcd-1234.ngrok.io')
```

Notes & debugging tips
----------------------
- Confirm the app uses `http://` for local testing. Mixed HTTPS vs HTTP can cause failures on some devices/browsers.
- Check app logs for the dev-only request dump `API Request =>` (we added this logging in `src/qf/lib/api.ts`). It prints URL, method, headers, and body — paste that output if issues persist.
- On Android devices, use `adb logcat` to see device logs while reproducing the failure:

```powershell
adb logcat | findstr "API Request" 
# or monitor React Native logs: ReactNativeJS:V
```

- Backend logs (uvicorn) show incoming requests with timestamps — match them to the app logs to confirm whether the request reaches the server.
- Common problems:
  - Wrong scheme (app using https while backend is http)
  - Firewall blocking inbound
  - Phone not on same network
  - Wrong port or IP typed in the app

If still failing
----------------
- Paste the exact `API Request =>` log line from the app and any fetch error stack.
- Paste the backend uvicorn log line for that request (timestamped).

If you'd like
------------
I can add a small in-app debug screen (input + save) so your friend can type the backend URL on the phone and persist it — say the word and I'll add it.

---
File created at: `src/qf/config/referThis.md` — use it as a handheld guide when handing the project to another developer or tester.
