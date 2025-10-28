if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Run the Image Translator Pipeline")
    parser.add_argument("--host", default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=8080, help="Port number")
    parser.add_argument("--reload", action="store_true", help="Enable autoreload")
    args = parser.parse_args()

    uvicorn.run(
        "image_translator.streaming_pipeline:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
