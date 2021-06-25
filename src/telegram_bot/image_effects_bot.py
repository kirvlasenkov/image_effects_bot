# -*- coding: utf-8 -*-
import logging
from concurrent.futures import ThreadPoolExecutor

# from telegram_token import token
from io import BytesIO
from time import sleep

from model import StyleTransferModel
from telegram.ext import CommandHandler, ConversationHandler, Filters, MessageHandler, Updater
from telegram.ext.dispatcher import run_async

context = None
error = None

pool = ThreadPoolExecutor(1)

# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

model = StyleTransferModel()
content_images_files = {}
style_images_files = {}
CONTENT, STYLE = range(2)


def send_prediction_on_photo(content_image_stream, style_image_stream, chat_id, context):

    # process images
    future = pool.submit(model.transfer_style, content_image_stream, style_image_stream)
    output = future.result()

    # send back
    output_stream = BytesIO()
    output.save(output_stream, format="PNG")
    output_stream.seek(0)
    # context.bot.send_photo(chat_id, photo=output_stream)
    del content_images_files[chat_id]
    del style_images_files[chat_id]
    print("Sent Photo to user")


@run_async
def handle_message(self, bot, update):
    context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=(
            "Hi, my name is Image Effects. I can add various effects to the photo, as well as"
            " increase the quality of the image!"
        ),
    )
    context.bot.send_message(
        chat_id=update.effective_chat.id, text="What kind of processing are you interested in?"
    )


@run_async
def get_content_image(update, context):
    chat_id = update.message.chat_id
    print("Got image from {}".format(chat_id))
    image_info = update.message.photo[-1]
    image_file = context.bot.get_file(image_info)
    content_image_stream = BytesIO()
    image_file.download(out=content_image_stream)
    content_images_files[chat_id] = content_image_stream
    context.bot.send_message(chat_id=update.effective_chat.id, text="Please send style image")
    return STYLE


if __name__ == "__main__":
    TOKEN = "1808716289:AAFp1FjPnsd3QgF7VscBglvfymqYA1q-qF8"
    updater = Updater(token=TOKEN, use_context=True)

    # updater.dispatcher.add_handler(style_transfer_handler)
    updater.dispatcher.add_error_handler(error)
    updater.start_polling()
