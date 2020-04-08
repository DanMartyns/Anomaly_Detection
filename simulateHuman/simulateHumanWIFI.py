import time
import json
import requests
from datetime import datetime,timedelta

##################################################
# Acessos desmedidos para simular uma grande  
# quantidade de dados por WI-FI
##################################################

dic = []

# leitura do histórico de pesquisa do google chrome
with open('chrome_history.json') as json_file:
    data = json.load(json_file)
    for p in data:
        dt = datetime.strptime(p['lastVisitTimeUTC'], "%Y-%m-%dT%H:%M:%S.%fZ")
        # cada linha representa o titulo, quando foi feito a chamada, o url da chamada
        dic.append([p['title'],dt, p['url']])  

dic.reverse()

while True:
    # para cada actividade
    for index,item in enumerate(dic[1:]):
        try:
            print("Current Activity :",dic[index][0])
            # intervalo entre a proxima activade e actividade currente
            interval = (item[1] - dic[index][1]).total_seconds()
            # print do tempo que falta até à próxima actividade
            print("Next Activity in",str(timedelta(seconds=interval)))
            # todo o intervalo que que seja mais que uma hora é reduzido a 0 e a actividade é desempenhada logo
            if interval > 86400:
                interval = 0
            # Get da actividade para efeitos de simulação do que foi feito no histórico do chrome
            r = requests.get(item[2])
            # Confirmação que tudo correu bem
            print("Get Status: ",r.status_code,"\n ---")
            # sleep por igual período até à próxima actividade
            if "Youtube" in dic[index][0]:
                time.sleep(5)
        except Exception as e:
            print(e)
