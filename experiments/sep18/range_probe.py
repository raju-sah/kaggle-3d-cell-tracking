import subprocess,sys
subprocess.run([sys.executable,'-m','pip','install','-q','remotezip'],check=True)
from kaggle.api.kaggle_api_extended import KaggleApi,ApiDownloadDataFilesRequest
from remotezip import RemoteZip
from pathlib import Path
api=KaggleApi();api.authenticate()
with api.build_kaggle_client() as client:
 r=ApiDownloadDataFilesRequest();r.competition_name='biohub-cell-tracking-during-development'
 response=client.competitions.competition_api_client.download_data_files(r)
 url=response.url
 with RemoteZip(url) as z:
  infos=z.infolist()
  print('entries',len(infos),'compressed_GB',sum(i.compress_size for i in infos)/1e9)
  print('first20',[i.filename for i in infos[:20]])
  train=[i.filename for i in infos if i.filename.startswith('train/')]
  print('train first20',train[:20])
 Path('/content/biohub/archive_url').write_text(url)
 Path('/content/biohub/archive_url').chmod(0o600)
