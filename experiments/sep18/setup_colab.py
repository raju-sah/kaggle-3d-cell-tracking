from pathlib import Path
import subprocess, os, json
key = Path('/content/kaggle_access_token')
dst = Path.home()/'.kaggle/access_token'
dst.parent.mkdir(exist_ok=True)
if key.exists():
    key.replace(dst)
    dst.chmod(0o600)
root=Path('/content/biohub'); root.mkdir(exist_ok=True)
jobs={
 'competition':['kaggle','competitions','download','-c','biohub-cell-tracking-during-development','-p',str(root/'competition'),'--quiet'],
 'support':['kaggle','datasets','download','-d','pilkwang/biohub-tracking-support-pack-50ep-v1','-p',str(root/'support'),'--unzip','--quiet'],
 'secondary':['kaggle','datasets','download','-d','pilkwang/biohub-temporal-unet3d-seed314159-v1','-p',str(root/'secondary'),'--unzip','--quiet'],
 'deepcenter':['kaggle','datasets','download','-d','pilkwang/biohub-deepcenter-unet3d-center-prior-v1','-p',str(root/'deepcenter'),'--unzip','--quiet'],
}
for name,cmd in jobs.items():
 with (root/f'{name}_download.log').open('w') as out:
  p=subprocess.Popen(cmd,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
 print(name,p.pid)
