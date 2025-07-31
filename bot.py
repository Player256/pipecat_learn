#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import argparse
import os

import aiohttp
from dotenv import load_dotenv
from loguru import logger

from pipecat.frames.frames import EndFrame, TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.services.rime.tts import RimeHttpTTSService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.network.fastapi_websocket import FastAPIWebsocketParams
from pipecat.transports.network.small_webrtc import SmallWebRTCTransport
from pipecat.transports.services.daily import DailyParams
from pipecat.audio.vad.silero import SileroVADAnalyzer

load_dotenv(override=True)

# We store functions so objects (e.g. SileroVADAnalyzer) don't get
# instantiated. The function will be called when the desired transport gets
# selected.
transport_params = {
    "daily": lambda: DailyParams(audio_out_enabled=True),
    "twilio": lambda: FastAPIWebsocketParams(audio_out_enabled=True),
    "webrtc": lambda: TransportParams(audio_out_enabled=True),
}


async def run_example(
    webrtc_connection
):
    logger.info(f"Starting bot")

    transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),  # Uncomment to enable VAD
        )
    )

    # Create an HTTP session
    async with aiohttp.ClientSession() as session:
        tts = RimeHttpTTSService(
            api_key=os.getenv("RIME_API_KEY", ""),
            voice_id="rex",
            aiohttp_session=session,
        )

        task = PipelineTask(Pipeline([tts, transport.output()]))

        # Register an event handler so we can play the audio when the client joins
        @transport.event_handler("on_client_connected")
        async def on_client_connected(transport, client):
            await task.queue_frames([TTSSpeakFrame(f"Hello there!"), EndFrame()])

        runner = PipelineRunner(handle_sigint=False)

        await runner.run(task)


# if __name__ == "__main__":
#     from pipecat.examples.run import main

#     main(run_example, transport_params=transport_params)
