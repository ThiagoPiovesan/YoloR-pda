from index import *

class UserBot():
  def __init__(self, user_token, user_id):
    self.user_token = user_token
    self.user_id = user_id
    self.bot = telebot.TeleBot(user_token)  # creating a instance
    bot = self.bot

  def send_alert(self, msg):
    self.bot.send_message(self.user_id, msg)

