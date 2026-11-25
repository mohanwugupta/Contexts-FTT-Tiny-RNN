# Cloud native trial v1 results

Start with [REPORT.md](REPORT.md), [RESULT.json](RESULT.json), and [LATEST_LAUNCH.json](LATEST_LAUNCH.json). The trial completed all 16 target jobs. The registered scientific status is `background_inconclusive`.

The verified export contains 361 files. To reconstruct and verify the original ZIP, run these commands from this directory:

```bash
cat results.zip.part-00 results.zip.part-01 > results.zip
sha256sum -c results.zip.sha256
```
