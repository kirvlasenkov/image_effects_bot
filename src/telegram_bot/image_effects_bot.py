# -*- coding: utf-8 -*-
import logging
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from time import sleep

from image_effects_model import StyleTransferModel
from telegram.ext import CommandHandler, ConversationHandler, Filters, MessageHandler, Updater
from telegram.ext.dispatcher import run_async

pool = ThreadPoolExecutor(1)

# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

image_effects_model = StyleTransferModel()
content_images_files = {}
style_images_files = {}
CONTENT, STYLE = range(2)


def send_prediction_on_photo(content_image_stream, style_image_stream, chat_id, context):

    # process images
    future = pool.submit(
        image_effects_model.transfer_style, content_image_stream, style_image_stream
    )
    output = future.result()

    # send back
    output_stream = BytesIO()
    output.save(output_stream, format="PNG")
    output_stream.seek(0)
    context.bot.send_photo(chat_id, photo=output_stream)
    del content_images_files[chat_id]
    del style_images_files[chat_id]
    print("Sent Photo to user")


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


@run_async
def get_style_image(update, context):
    chat_id = update.message.chat_id
    print("Got image from {}".format(chat_id))
    image_info = update.message.photo[-1]
    image_file = context.bot.get_file(image_info)
    style_image_stream = BytesIO()
    image_file.download(out=style_image_stream)
    style_images_files[chat_id] = style_image_stream
    context.bot.send_message(
        chat_id=update.effective_chat.id, text="Please wait while processing images..."
    )
    send_prediction_on_photo(
        content_images_files[chat_id], style_images_files[chat_id], chat_id, context
    )
    context.bot.send_message(chat_id=update.effective_chat.id, text="Please send content image")
    return CONTENT


@run_async
def start(update, context):
    context.bot.send_message(chat_id=update.effective_chat.id, text="Please send content image")
    return CONTENT


@run_async
def cancel(update, context):
    context.bot.send_message(chat_id=update.effective_chat.id, text="Canceled")
    return ConversationHandler.END


style_transfer_handler = ConversationHandler(
    entry_points=[CommandHandler("start", start)],
    states={
        CONTENT: [MessageHandler(Filters.photo, get_content_image, pass_user_data=True)],
        STYLE: [MessageHandler(Filters.photo, get_style_image, pass_user_data=True)],
    },
    fallbacks=[CommandHandler("cancel", cancel)],
)


@run_async
def error(context, update, error):
    """Log Errors caused by Updates."""
    logger.warning('Update "%s" caused error "%s"', update, error)


if __name__ == "__main__":
    TOKEN = ""
    updater = Updater(token=TOKEN, use_context=True)

    updater.dispatcher.add_handler(style_transfer_handler)
    updater.dispatcher.add_error_handler(error)
    updater.start_polling()
