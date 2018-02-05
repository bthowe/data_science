import json
import requests
import numpy as np
import pandas as pd

header = {'Content-Type': 'application/json', 'Accept': 'application/json'}


np.random.seed(4)
df = pd.concat(
    [
        pd.DataFrame(np.random.uniform(0, 1, size=(20, 3)), columns=['one', 'two', 'three']),
    ], axis=1
)

data = df.to_json(orient='records')

resp = requests.post("http://0.0.0.0:8000/predict", headers=header)
# resp = requests.post("http://0.0.0.0:8000/predict", data=json.dumps(data), headers=header)

print(resp.status_code)
print(resp.json())
