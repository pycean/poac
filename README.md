## Installation instructions

use python3.6

```shell
pip install -r requirements
```

## Run an experiment 

for red
```shell
cd POAC-master
python3 src/main.py --config=qmix --env-config=bq --seed_y=1 
python3 src/main.py --config=qmix --env-config=bqhard1 --seed_y=1 
```
for blue
```shell
cd POAC-master
python3 src/main.py --config=qmix --env-config=bq1 --seed_y=1 
python3 src/main.py --config=qmix --env-config=bq1hard1 --seed_y=1 
```



## License

Code licensed under the Apache License v2.0
#   p o a c  
 