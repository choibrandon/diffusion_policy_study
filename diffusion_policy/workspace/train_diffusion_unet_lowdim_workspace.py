if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent) #상위 3단계 폴더를 ROOT_DIR로 지정
    sys.path.append(ROOT_DIR) # 방금 전 위의 경로를 python path로 지정
    os.chdir(ROOT_DIR) # 현재 작업 디렉토리를 workspace를 ROOT_DIR로 변경 즉, 코드 실행시 경로 문제가 발생하지 않도록 보장 
# 스크립트의 실행 진입점을 맨앞으로 놓은 이유는 모듈을 임포트 할 때, 보통 sys.path.append(ROOT_DIR)같은 거를 해야 모듈을 잘 찾을 수 있음 그래서 미리 
# 하는 거임. 그래야 import diffusion_policy 와 같이 아래 것들을 잘 찾을 수 있음. 
# 새로운 사실, os.chdir(ROOT_DIR)이 먼저 실행하면 어떤 경로에서든지 항상 프로젝트 루트에서 실행 되도록 보장.


import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import numpy as np
import random
import wandb
import tqdm
import shutil

from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.diffusion_unet_lowdim_policy import DiffusionUnetLowdimPolicy
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusers.training_utils import EMAModel

OmegaConf.register_new_resolver("eval", eval, replace=True)

# %%
class TrainDiffusionUnetLowdimWorkspace(BaseWorkspace): #클래스 선언 하고 상위 클래스 BaseWorkspace 클래스 기능을 상속
    include_keys = ['global_step', 'epoch'] #이 클래스에서 저장하거나 로깅할 중요한 변수를 지정하는 리스트임
    #baseworkspace에서 데이터를 저장할 때, 이 키들을 저장하거나 불러오는 데 사용, 이건 체크포인트 저장할 때, 이변수들을 반드시 저장하도록 지정하는 역할

    def __init__(self, cfg: OmegaConf, output_dir=None): #hydra 
        super().__init__(cfg, output_dir=output_dir)

        # set seed 랜덤 시드를 고정하면, 같은 입력에 대해 동일한 결과가 나옴(재현 가능성 확보)
        seed = cfg.training.seed 
        torch.manual_seed(seed) #pytorch 연산의 랜덤 시드 고정
        np.random.seed(seed) # numpy의 랜덤 시드 고정
        random.seed(seed) # python 기본 random 모듈 시드

        # configure model
        self.model: DiffusionUnetLowdimPolicy #self.model이 DiffusionUnetPolicy 타입의 객체임을 명시 아직은 인스턴스 화 하지않음
        self.model = hydra.utils.instantiate(cfg.policy) #모델 인스턴스화 하이드라에서 

        self.ema_model: DiffusionUnetLowdimPolicy = None #ema(지수이동평균)모델은 훈련 중 모델의 가중치를 부드럽게 보정하여 더 안정적인 모델을 만드는 기법이다.
        if cfg.training.use_ema: #cfg.training.use_ema=True 이면 사용함
            self.ema_model = copy.deepcopy(self.model) #실제학습에는 사용하지 않고 더 나은 예측을 위해 따로 관리됨.
        #ema 모델을 사용하면 가중치를 천천히 업데이트 하여 과적합 방지 효과가 있다. 

        # configure training state
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters())
        # 훈련상태 변수 초기화
        self.global_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg) #여기서도 cfg 를 딥 카피함 기존 설정을 변경없이 유지하기 유하여
        #cfg를 변경해도 self.cfg를 변하지 않도록
        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        # configure dataset 데이터셋 가져오고
        dataset: BaseLowdimDataset #추후확인
        dataset = hydra.utils.instantiate(cfg.task.dataset) #아마도 task.dataset에 있는 클래스를 인스턴스화 하는 듯 task는 pusht_lowdim 같음
        assert isinstance(dataset, BaseLowdimDataset) #인스턴스화 되었는지 확인 dataset이게 BaseLowdimDataset의 인스턴스인지 확인
        train_dataloader = DataLoader(dataset, **cfg.dataloader) #데이터 로더를 만들어줌 그리고 **cfg.dataloader는 cfg파일에 있는 dataloader의 설정을 가져옴
        normalizer = dataset.get_normalizer()

        # configure validation dataset 벨리데이션
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        self.model.set_normalizer(normalizer) #ema모델에도 normalizer를 설정 normalizer는 데이터셋의 평균과 표준편차를 저장하는 객체임
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # configure lr scheduler 학습률 스케쥴러
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step-1 
            #초기값 설정, pytorch의 많은 lr 스케줄러는 기본적으로 last_epoch=-1로 설정, 훈련을 처음 시작하는 경우 self.global_step은 0일텐데, 0-1 = -1이 되어 초기값과 맞아 떨어진다.
        )

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)

        # configure env runner
        env_runner: BaseLowdimRunner #추후확인
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=self.output_dir)
        assert isinstance(env_runner, BaseLowdimRunner)

        # configure logging
        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.logging
        )
        wandb.config.update(
            {
                "output_dir": self.output_dir,
            }
        )

        # configure checkpoint
        topk_manager = TopKCheckpointManager( #추후확인
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        # device transfer
        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        # save batch for sampling
        train_sampling_batch = None

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                step_log = dict()
                # ========= train for this epoch ==========
                train_losses = list()
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        # device transfer
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        if train_sampling_batch is None:
                            train_sampling_batch = batch

                        # compute loss
                        raw_loss = self.model.compute_loss(batch)
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        loss.backward()

                        # step optimizer
                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()
                        
                        # update ema
                        if cfg.training.use_ema:
                            ema.step(self.model)

                        # logging
                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            'train_loss': raw_loss_cpu,
                            'global_step': self.global_step,
                            'epoch': self.epoch,
                            'lr': lr_scheduler.get_last_lr()[0]
                        }

                        is_last_batch = (batch_idx == (len(train_dataloader)-1))
                        if not is_last_batch:
                            # log of last step is combined with validation and rollout
                            wandb_run.log(step_log, step=self.global_step)
                            json_logger.log(step_log)
                            self.global_step += 1

                        if (cfg.training.max_train_steps is not None) \
                            and batch_idx >= (cfg.training.max_train_steps-1):
                            break
                
                # at the end of each epoch
                # replace train_loss with epoch average
                train_loss = np.mean(train_losses)
                step_log['train_loss'] = train_loss

                # ========= eval for this epoch ==========
                policy = self.model
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()

                # run rollout
                if (self.epoch % cfg.training.rollout_every) == 0:
                    runner_log = env_runner.run(policy)
                    # log all
                    step_log.update(runner_log)

                # run validation
                if (self.epoch % cfg.training.val_every) == 0:
                    with torch.no_grad():
                        val_losses = list()
                        with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}", 
                                leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                                loss = self.model.compute_loss(batch)
                                val_losses.append(loss)
                                if (cfg.training.max_val_steps is not None) \
                                    and batch_idx >= (cfg.training.max_val_steps-1):
                                    break
                        if len(val_losses) > 0:
                            val_loss = torch.mean(torch.tensor(val_losses)).item()
                            # log epoch average validation loss
                            step_log['val_loss'] = val_loss

                # run diffusion sampling on a training batch
                if (self.epoch % cfg.training.sample_every) == 0:
                    with torch.no_grad():
                        # sample trajectory from training set, and evaluate difference
                        batch = train_sampling_batch
                        obs_dict = {'obs': batch['obs']}
                        gt_action = batch['action']
                        
                        result = policy.predict_action(obs_dict)
                        if cfg.pred_action_steps_only:
                            pred_action = result['action']
                            start = cfg.n_obs_steps - 1
                            end = start + cfg.n_action_steps
                            gt_action = gt_action[:,start:end]
                        else:
                            pred_action = result['action_pred']
                        mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                        # log
                        step_log['train_action_mse_error'] = mse.item()
                        # release RAM
                        del batch
                        del obs_dict
                        del gt_action
                        del result
                        del pred_action
                        del mse
                
                # checkpoint
                if (self.epoch % cfg.training.checkpoint_every) == 0:
                    # checkpointing
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()

                    # sanitize metric names
                    metric_dict = dict()
                    for key, value in step_log.items():
                        new_key = key.replace('/', '_')
                        metric_dict[new_key] = value
                    
                    # We can't copy the last checkpoint here
                    # since save_checkpoint uses threads.
                    # therefore at this point the file might have been empty!
                    topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                    if topk_ckpt_path is not None:
                        self.save_checkpoint(path=topk_ckpt_path)
                # ========= eval end for this epoch ==========
                policy.train()

                # end of epoch
                # log of last step is combined with validation and rollout
                wandb_run.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainDiffusionUnetLowdimWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
