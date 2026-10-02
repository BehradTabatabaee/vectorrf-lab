"""Deterministic VectorRF Lab browser smoke test.

Loads the real HTML/CSS/JavaScript in Chromium while mocking API/WebSocket
transport. This keeps the UI release gate runnable without a background server.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

from playwright.sync_api import sync_playwright


ROOT = Path(__file__).resolve().parents[2]
WEB = ROOT / "backend" / "vectorrf" / "web"
BROWSER = os.environ.get("VECTORRF_BROWSER_PATH")


def main() -> None:
    html = (WEB / "index.html").read_text()
    html = re.sub(r'<link[^>]+styles\.css[^>]*>', '', html)
    html = re.sub(r'<script[^>]+src="/assets/(?:app|studio|workbench)\.js"[^>]*></script>', '', html)

    with sync_playwright() as p:
        launch_args = {"headless": True, "args": ["--no-sandbox"]}
        if BROWSER:
            launch_args["executable_path"] = BROWSER
        browser = p.chromium.launch(**launch_args)
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        errors: list[str] = []
        page.on("pageerror", lambda exc: errors.append(str(exc)))
        page.set_content(html)
        page.add_style_tag(path=str(WEB / "styles.css"))
        page.add_init_script(
            """
            class FakeWS {
              static OPEN=1; static CONNECTING=0;
              constructor(){this.readyState=0;setTimeout(()=>{this.readyState=1;this.onopen&&this.onopen();},0)}
              send(){} close(){this.readyState=3;}
            }
            window.WebSocket=FakeWS;
            window.fetch=async function(url,opts){
              const u=String(url);let data={};
              if(u.includes('/api/health'))data={ok:true,version:'1.2.0',max_spectrum_span:10000000};
              else if(u.includes('/api/catalog'))data={modes:['tx','spectrum','rx','blind','radar']};
              else if(u.includes('/api/studio/extensions'))data={extensions:[]};
              else if(u.includes('/api/measurements'))data={
                time_s:[0,1e-6,2e-6],i:[1,0,-1],q:[0,1,0],amplitude:[1,1,1],
                phase_deg:[0,90,180],frequency_hz:[0,1000,1000],valid:[true,true,true],
                sample_rate:1000000,capture_start_s:0,capture_duration_s:2e-6,rms:1,peak:1,
                quality:null,source_revision:'fake',capture_id:0
              };
              return new Response(JSON.stringify(data),{status:200,headers:{'Content-Type':'application/json'}});
            };
            """
        )
        page.add_script_tag(path=str(WEB / "app.js"))
        page.add_script_tag(path=str(WEB / "studio.js"))
        page.add_script_tag(path=str(WEB / "workbench.js"))
        page.wait_for_timeout(250)

        assert page.locator('.help-link').count() == 1
        assert page.locator('.help-link').get_attribute('href') == '/help'

        page.evaluate("switchMode('spectrum')")
        page.wait_for_timeout(50)
        page.eval_on_selector(
            '[data-bind="spectrum.span_hz"]',
            '(e)=>{e.value="50000000";e.dispatchEvent(new Event("change",{bubbles:true}))}',
        )
        page.wait_for_timeout(30)
        assert page.evaluate('state.spectrum.span_hz') == 10_000_000
        assert page.input_value('[data-bind="spectrum.span_hz"]') == '10000000'
        bounds = page.evaluate('spectrumViewBounds()')
        assert bounds['left'] == -5_000_000 and bounds['right'] == 5_000_000

        # Feed a 1 MS/s source into a 10 MHz requested view. Rendering must keep
        # the 10 MHz axis and tolerate the narrower source-Nyquist data.
        page.evaluate(
            """()=>{
              state.live.freq=Array.from({length:1025},(_,i)=>-500000+i*1000000/1024);
              state.live.psd=state.live.freq.map(f=>-80+35*Math.exp(-((f/80000)**2)));
              state.live.waterfallRow=Float32Array.from(state.live.psd);
              state.live.metrics={rbw_actual_hz:2000,peak_hz:0,peak_db:-45};
              drawSpectrumPage();
            }"""
        )

        # Verify the release zero-span cadence without relying on wall-clock API
        # responses. startMode installs an immediate refresh plus a 12 Hz timer.
        count = page.evaluate(
            """async()=>{
              stopModeTimers();state.mode='spectrum';state.specPage='zero';window.__zcount=0;
              refreshMeasurements=async()=>{window.__zcount++};startMode();
              await new Promise(r=>setTimeout(r,360));stopModeTimers();return window.__zcount;
            }"""
        )
        assert count >= 4, count
        assert errors == [], errors
        browser.close()

    print({
        'passed': True,
        'span_clamped_hz': 10_000_000,
        'view_left_hz': -5_000_000,
        'view_right_hz': 5_000_000,
        'zero_span_calls_360ms': count,
        'page_errors': errors,
    })


if __name__ == '__main__':
    main()
