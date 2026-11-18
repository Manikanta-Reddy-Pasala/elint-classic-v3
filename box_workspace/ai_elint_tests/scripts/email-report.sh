#!/bin/bash
# email-report.sh
# Sends an Allure emailable report as an HTML email attachment (gzip-compressed).
# All configuration is passed via environment variables.

python3 - <<'EOF'
import smtplib
import os
import re
import gzip
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

report_url      = os.environ.get('REPORT_URL', '')
project_id      = os.environ.get('PROJECT_TEST_ID', '')
pipeline_url    = os.environ.get('CI_PIPELINE_URL', '')
pipeline_id     = os.environ.get('CI_PIPELINE_ID', '')
pipeline_source = os.environ.get('CI_PIPELINE_SOURCE', '')
project_name    = os.environ.get('CI_PROJECT_NAME', '')
commit_sha      = os.environ.get('CI_COMMIT_SHORT_SHA', '')
commit_branch   = os.environ.get('CI_COMMIT_REF_NAME', '')
is_master       = commit_branch == 'master'

# Safety-net limit for the *compressed* attachment (bytes). If exceeded, the
# attachment is dropped and only the link-only summary email is sent, so the
# notification never fails because of SMTP size limits.
MAX_ATTACH_BYTES = int(os.environ.get('MAX_EMAIL_ATTACH_BYTES', str(9 * 1024 * 1024)))  # 9 MB default

if is_master:
    recipients = os.environ.get('ALLURE_REPORT_RECIPIENTS', '').split(',')
else:
    recipients = [os.environ.get('GITLAB_USER_EMAIL', os.environ.get('ALLURE_REPORT_RECIPIENTS', ''))]

with open('allure-emailable-report.html', 'rb') as f:
    html_bytes = f.read()

# Gzip-compress the emailable report (HTML compresses very well).
compressed_bytes = gzip.compress(html_bytes, compresslevel=9)
print(f"Emailable report: {len(html_bytes)} bytes raw -> {len(compressed_bytes)} bytes gzipped")

def extract_stat(label_class, content):
    thead_headers = re.findall(r'<th[^>]*class="([^"]*?)"[^>]*>(.*?)</th>', content, re.DOTALL)
    col_index = None
    for i, (cls, text) in enumerate(thead_headers):
        if label_class in cls:
            col_index = i
            break
    if col_index is None:
        return 'N/A'
    tbody_values = re.findall(r'<tbody.*?>(.*?)</tbody>', content, re.DOTALL)
    if not tbody_values:
        return 'N/A'
    cells = re.findall(r'<th[^>]*>(.*?)</th>', tbody_values[0], re.DOTALL)
    if col_index < len(cells):
        return re.sub(r'<[^>]+>', '', cells[col_index]).strip()
    return 'N/A'

html_content = html_bytes.decode('utf-8', errors='replace')
total   = extract_stat('table-active',  html_content)
passed  = extract_stat('table-success', html_content)
failed  = extract_stat('table-danger',  html_content)
broken  = extract_stat('table-warning', html_content)
skipped = extract_stat('table-light',   html_content)

attachment_skipped = len(compressed_bytes) > MAX_ATTACH_BYTES
attachment_note = (
    "<br><p><b>⚠️ The compressed report attachment was skipped</b> because it exceeded "
    f"the {MAX_ATTACH_BYTES // (1024*1024)} MB limit. "
    f"View the full report here: <a href=\"{report_url}\">{report_url}</a></p>"
    if attachment_skipped
    else
    "<br><p>The HTML report is attached (gzip-compressed as <code>.html.gz</code>). "
    "Decompress it with <code>gunzip allure-report-{project_id}.html.gz</code> "
    "(or right-click -> Extract / 7-Zip / WinRAR) to open it in a browser.</p>"
).format(project_id=project_id) if attachment_skipped else (
    "<br><p>The HTML report is attached (gzip-compressed as <code>.html.gz</code>). "
    "Decompress it with <code>gunzip allure-report-{project_id}.html.gz</code> "
    "(or right-click -> Extract / 7-Zip / WinRAR) to open it in a browser.</p>".format(project_id=project_id)
)

html_body = f"""
<html><body style="font-family: Arial, sans-serif; font-size: 14px; color: #333;">
  <p>An Allure test report has been generated.</p>
  <table style="border-collapse: collapse;">
    <tr><td style="padding: 4px 12px 4px 0;"><b>Project</b></td><td>{project_name}</td></tr>
    <tr><td style="padding: 4px 12px 4px 0;"><b>Branch</b></td><td>{commit_branch}</td></tr>
    <tr><td style="padding: 4px 12px 4px 0;"><b>Commit</b></td><td>{commit_sha}</td></tr>
    <tr><td style="padding: 4px 12px 4px 0;"><b>Pipeline</b></td><td>#{pipeline_id} ({pipeline_source})</td></tr>
    <tr><td style="padding: 4px 12px 4px 0;"><b>Pipeline URL</b></td><td><a href="{pipeline_url}">{pipeline_url}</a></td></tr>
    <tr><td style="padding: 4px 12px 4px 0;"><b>Test ID</b></td><td>{project_id}</td></tr>
    <tr><td style="padding: 4px 12px 4px 0;"><b>Full Report</b></td><td><a href="{report_url}">{report_url}</a></td></tr>
  </table>
  <br>
  <table style="border-collapse: collapse; margin-top: 8px;">
    <tr><td style="padding: 4px 12px 4px 0;">Total tests run:</td>
        <td><b>{total}</b></td></tr>
    <tr><td style="padding: 4px 12px 4px 0;">Total tests passed:</td>
        <td><b><span style="color: #1a7a1a;">{passed}</span></b></td></tr>
    <tr><td style="padding: 4px 12px 4px 0;">Total tests failed:</td>
        <td><b><span style="color: #cc0000;">{failed}</span></b></td></tr>
    <tr><td style="padding: 4px 12px 4px 0;">Total tests broken:</td>
        <td><b><span style="color: #b87800;">{broken}</span></b></td></tr>
    <tr><td style="padding: 4px 12px 4px 0;">Total tests skipped:</td>
        <td><b><span style="color: #777777;">{skipped}</span></b></td></tr>
  </table>
  {attachment_note}
</body></html>
"""

msg = MIMEMultipart('mixed')
msg['Subject'] = f'Allure Test Report - {project_name} ({project_id})'
msg['From'] = os.environ.get('ALLURE_REPORT_EMAIL', '')
msg['To'] = ', '.join(recipients)

msg.attach(MIMEText(html_body, 'html'))

if not attachment_skipped:
    # Attach the gzip-compressed report (gzip content-type, .gz filename).
    attachment = MIMEBase('application', 'gzip')
    attachment.set_payload(compressed_bytes)
    encoders.encode_base64(attachment)
    attachment.add_header(
        'Content-Disposition',
        'attachment',
        filename=f'allure-report-{project_id}.html.gz'
    )
    msg.attach(attachment)
else:
    print(f"Skipping attachment: compressed size {len(compressed_bytes)} bytes "
          f"exceeds limit {MAX_ATTACH_BYTES} bytes")

with smtplib.SMTP(os.environ.get('SMTP_SERVER'), 25) as server:
    server.ehlo()
    server.sendmail(os.environ.get('ALLURE_REPORT_EMAIL', ''), recipients, msg.as_string())
    server.quit()

print('Email sent successfully' + (' (no attachment - over limit)' if attachment_skipped else ' with gzip-compressed HTML attachment'))
EOF