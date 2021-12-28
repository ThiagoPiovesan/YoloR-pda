## Introduction

This repository aims to store a telegram bot that has the function of receiving detections from an artificial intelligence and issuing an alert to the user via the telegram, thus alerting if it recognizes a jaguar or a person

## Requirements

Python 3.6+

## Installation

You should have git installed on your computer. You can download git <a href="https://git-scm.com/book/en/v2/Getting-Started-Installing-Git" target="_blank">here</a>.

Also you need to have pip installed. You can install pip by running the following commands:

```bash
$ python get-pip.py
```

After installing git choose a folder that you want to run the project and run the following comand:

```bash
$ git clone https://github.com/Jhow-Rambo/bot_jaguar.git
```

```bash
$ cd bot_jaguar
```

```bash
$ pip install -r /path/to/requirements.txt
```

After all this commands you should be able to run and use the bot.

## Testing the bot

To use the bot it's necessary to have the bot_jaguar folder inside of your project, so copy the bot_jaguar and past inside the folder. After copying the folder create a .env file inside bot_jaguar and past the following comands:

```
ADMIN_TOKEN= <TOKEN OF ADMIN BOT>
ADMIN_ID= <ID OF USER THAT THE TOKEN WILL COMUNICATE>

USER_TOKEN= <TOKEN OF USER BOT>
USER_ID = <ID OF USER THAT THE TOKEN WILL COMUNICATE>
```

After creating the .env file create a file outside bot_jaguar to import the Bot and set the token and id:

```
from bot_jaguar import AdminBot
from dotenv import load_dotenv
from threading import Thread
import os

load_dotenv()
token_admin = os.getenv('ADMIN_TOKEN')
admin_id = os.getenv('ADMIN_ID')
```

Create an instance for the bot passing the token and id:

```
tb = AdminBot(token_admin, admin_id)
```

Call some methods to test the bot:

```
tb.send_alert(detection='pessoa', accuracy='67%', img=None)

t1 = Thread(target=tb.alive)
t1.start()
```

The entire test code:

```
from bot_jaguar import AdminBot
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
```

Or you can just set the .env file and run the exampleOfUse.py to test the bot.

## Observations

Don't forget to read the AdminBot.py file to see all the functions and methods that can be called and used with the bot. Also, this porject has two instance of bot, one for the admin user that will be able to call some especial methods and another to commum users that can only receive messages.

## License

This project is licensed under the terms of the MIT license.