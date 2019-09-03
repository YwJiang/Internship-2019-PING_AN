port=9200

# kill es
lsof -i tcp:${port} | awk 'NR==2' | awk '{print $2}' | xargs kill -9

# run elastic search
sh ~/Downloads/elasticsearch-7.1.1/bin/elasticsearch &

# wait
sleep 15s

# run server
python3 src/server.py