import redis
from rsmq import RedisSMQ
from rsmq.consumer import RedisSMQConsumer

r = redis.Redis(port=6379, db=0)

queue_name = "entity"
mq = RedisSMQ(client=r, qname=queue_name)

# 清空消息队列
msg = mq.popMessage().execute()
while msg:
    msg = mq.popMessage().execute()

# 发送消息
message = 'hello'
mq.sendMessage().message(message).execute()


def processor(id, message, rc, ts):
    print(message)

# 接收消息
consumer = RedisSMQConsumer('gep_mq', processor, client=r)
consumer.run()
