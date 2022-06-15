import requests


def send_request(myobj):
    url = 'http://127.0.0.1:8000/predict'
    x = requests.post(url, json=myobj)
    return x.json()


if __name__ == '__main__':
    pass
