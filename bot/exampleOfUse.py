from AdminBot import AdminBot
from dotenv import load_dotenv
from threading import Thread
import os

load_dotenv()
token_admin = os.getenv('ADMIN_TOKEN')
admin_id = os.getenv('ADMIN_ID')

tb = AdminBot(token_admin, admin_id)

tb.send_alert(detection='pessoa', accuracy='67%', img=None)

t1 = Thread(target=tb.alive)
t1.start()
