"""
PDFPro — Server entry point.
עובד גם בפיתוח מקומי וגם ב-Render.
"""

import os
import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    debug = os.environ.get("DEBUG", "false").lower() == "true"

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=port,
        reload=debug,
        reload_dirs=["app"] if debug else None,
        workers=1,          # 1 worker = פחות RAM
        log_level="debug" if debug else "info",
    )