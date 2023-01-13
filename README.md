### ssh啟動指令
```sh
./ssl-proxy-linux-amd64 -from 192.168.10.112:9998 -to 192.168.10.112:8502
```

### ssh綁網域設定測試
```sh
export STREAMLIT_RUN_FILE_OR_URL=navigation.py
```

### 串流執行
```sh
python -m streamlit run navigation.py --server.port 9998 --server.enableCORS=false
```