"""
Usage:
Training:
python train.py --config-name=train_diffusion_lowdim_workspace
"""

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1) # 현재 표준 출력 stout 파일의 파일 디스크립터를 반환
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1) #현재 표준 에러 파일의 파일 디스크립터를 반환
# 파일 디스크립터는 os에서 파일을 다룰 때 사용하는 정수 값, python에서는 속성 접근 방식을 관리, 클래스의 속성의 읽기/ 쓰기/ 삭제 동작을 커스터마이징
import hydra
from omegaconf import OmegaConf
import pathlib
from diffusion_policy.workspace.base_workspace import BaseWorkspace

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)

@hydra.main(
    version_base=None, # hydra의 버전 설명 (None은 기본값)
    config_path=str(pathlib.Path(__file__).parent.joinpath( #보면 parent로 나와 있음
        'diffusion_policy','config')) # 설정 파일이 들어 있는 폴더 경로 
)
def main(cfg: OmegaConf):
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    OmegaConf.resolve(cfg)

    cls = hydra.utils.get_class(cfg._target_) #cfg.__target__ 값이 diffusion_policy.workspace.train_diffusion_unet_lowdim_workspace.TrainDiffusionUnetLowdimWorkspace 값을 인스턴스화하여 실행
    workspace: BaseWorkspace = cls(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
