call conda activate intelliface
python app.py
echo starting!...
timeout /t 5 /nobreak >nul
start "" "C:\Program Files (x86)\Microsoft\Edge\Application\msedge_proxy.exe" --profile-directory=Default --app-id=dknpifhceekpipgeechonedhlnddahhj --app-url=http://127.0.0.1:18080/ --app-launch-source=4

