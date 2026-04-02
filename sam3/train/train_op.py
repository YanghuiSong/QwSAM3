# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

import logging
import os
import random
import sys
import traceback
from argparse import ArgumentParser
from copy import deepcopy

import submitit
import torch
import torch.nn as nn

from hydra import compose, initialize_config_module
from hydra.utils import instantiate

from iopath.common.file_io import g_pathmgr
from omegaconf import OmegaConf

from sam3.train.utils.train_utils import makedir, register_omegaconf_resolvers
from tqdm import tqdm


os.environ["HYDRA_FULL_ERROR"] = "1"


class SlurmEvent:
    QUEUED = "QUEUED"
    START = "START"
    FINISH = "FINISH"
    JOB_ERROR = "JOB_ERROR"
    SLURM_SIGNAL = "SLURM_SIGNAL"


def validate_data_loader(data_loader, max_batches=3):
    """验证数据加载器的输出，检查NaN和数值范围"""
    print("=" * 60)
    print("开始数据验证...")
    print("=" * 60)
    
    try:
        for i, batch in enumerate(data_loader):
            if i >= max_batches:
                break
                
            print(f"验证批次 {i}:")
            
            # 检查图像数据
            if 'images' in batch:
                images = batch['images']
                if torch.is_tensor(images):
                    print(f"  图像 - 范围: [{images.min().item():.6f}, {images.max().item():.6f}]")
                    print(f"  图像 - 均值: {images.mean().item():.6f}, 标准差: {images.std().item():.6f}")
                    
                    if torch.isnan(images).any():
                        print("  ❌ 错误: 图像中包含NaN!")
                        return False
                    if torch.isinf(images).any():
                        print("  ❌ 错误: 图像中包含Inf!")
                        return False
                    
                    # 检查图像是否在合理范围内
                    if images.min() < -10 or images.max() > 10:
                        print(f"  ⚠️  警告: 图像范围异常，可能未正确归一化")
            
            # 检查所有张量数据
            for key, value in batch.items():
                if torch.is_tensor(value):
                    if torch.isnan(value).any():
                        print(f"  ❌ 错误: {key} 中包含NaN!")
                        return False
                    if torch.isinf(value).any():
                        print(f"  ❌ 错误: {key} 中包含Inf!")
                        return False
                    
                    # 打印关键数据的统计信息
                    if key in ['boxes', 'points', 'masks']:
                        print(f"  {key} - 范围: [{value.min().item():.6f}, {value.max().item():.6f}]")
            
            print(f"  ✅ 批次 {i} 数据检查通过")
        
        print("=" * 60)
        print("✅ 所有数据检查通过!")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"❌ 数据验证过程中出现异常: {e}")
        traceback.print_exc()
        return False


def validate_config(cfg):
    """验证配置参数是否合理"""
    print("=" * 60)
    print("开始配置验证...")
    print("=" * 60)
    
    warnings = []
    errors = []
    
    try:
        # 检查学习率设置
        if hasattr(cfg, 'scratch'):
            scratch = cfg.scratch
            lr_scale = scratch.get('lr_scale', 0.1)
            
            if lr_scale > 0.1:
                warnings.append(f"lr_scale={lr_scale} 可能过高，建议 <= 0.1")
            
            # 计算实际学习率
            if hasattr(scratch, 'lr_transformer'):
                actual_lr = scratch.lr_transformer
                if actual_lr > 1e-4:
                    warnings.append(f"transformer学习率 {actual_lr} 可能过高")
        
        # 检查梯度裁剪设置
        if hasattr(cfg.trainer, 'optim') and hasattr(cfg.trainer.optim, 'gradient_clip'):
            grad_clip = cfg.trainer.optim.gradient_clip
            max_norm = grad_clip.get('max_norm', 0.1)
            
            if max_norm < 0.5:
                warnings.append(f"梯度裁剪阈值 {max_norm} 可能过小，建议 >= 0.5")
        
        # 检查混合精度设置
        if hasattr(cfg.trainer.optim, 'amp'):
            amp_cfg = cfg.trainer.optim.amp
            if amp_cfg.get('enabled', False):
                amp_dtype = amp_cfg.get('amp_dtype', 'float16')
                if amp_dtype == 'bfloat16':
                    warnings.append("bfloat16混合精度可能在某些硬件上不稳定")
        
        # 检查模型配置
        if hasattr(cfg.trainer, 'model'):
            model_cfg = cfg.trainer.model
            if model_cfg.get('eval_mode', True):
                warnings.append("模型处于eval_mode，可能影响训练效果")
        
        # 输出验证结果
        if warnings:
            print("⚠️  配置警告:")
            for warning in warnings:
                print(f"   - {warning}")
        
        if errors:
            print("❌ 配置错误:")
            for error in errors:
                print(f"   - {error}")
            return False
        else:
            print("✅ 配置验证通过!")
            return True
            
    except Exception as e:
        print(f"❌ 配置验证过程中出现异常: {e}")
        return False


def check_model_parameters(model):
    """检查模型参数是否存在NaN或Inf"""
    print("=" * 60)
    print("检查模型参数...")
    print("=" * 60)
    
    try:
        nan_params = []
        inf_params = []
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                if torch.isnan(param).any():
                    nan_params.append(name)
                if torch.isinf(param).any():
                    inf_params.append(name)
        
        if nan_params:
            print("❌ 发现包含NaN的参数:")
            for name in nan_params[:5]:  # 只显示前5个
                print(f"  - {name}")
            return False
        
        if inf_params:
            print("❌ 发现包含Inf的参数:")
            for name in inf_params[:5]:  # 只显示前5个
                print(f"  - {name}")
            return False
        
        print("✅ 模型参数检查通过!")
        return True
        
    except Exception as e:
        print(f"❌ 模型参数检查失败: {e}")
        return False


def setup_numerical_stability():
    """设置数值稳定性相关的环境变量和配置"""
    print("=" * 60)
    print("设置数值稳定性配置...")
    print("=" * 60)
    
    # 设置PyTorch数值稳定性选项
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # 设置环境变量
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
    os.environ["TORCH_USE_CUDA_DSA"] = "0"
    
    print("✅ 数值稳定性配置完成")


def handle_custom_resolving(cfg):
    # We'll resolve the config here, so we can catch mistakes early.
    # However, we need to pass the un-resolved config to the launcher
    # (because DVC resolving needs to be done on the node it will run on)
    # First, do a copy without triggering resolving
    cfg_resolved = OmegaConf.to_container(cfg, resolve=False)
    cfg_resolved = OmegaConf.create(cfg_resolved)
    return cfg_resolved


def single_proc_run(local_rank, main_port, cfg, world_size):
    """Single GPU process"""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(main_port)
    os.environ["RANK"] = str(local_rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    
    try:
        register_omegaconf_resolvers()
    except Exception as e:
        logging.info(e)

    # 设置数值稳定性
    setup_numerical_stability()
    
    # 实例化trainer
    trainer = instantiate(cfg.trainer, _recursive_=False)
    
    # 检查模型参数
    if hasattr(trainer, 'model'):
        if not check_model_parameters(trainer.model):
            print("❌ 模型参数检查失败，退出训练")
            return
    
    # 验证数据加载器
    if hasattr(trainer, 'data_loader') and trainer.data_loader is not None:
        if not validate_data_loader(trainer.data_loader):
            print("❌ 数据验证失败，退出训练")
            return
    else:
        print("⚠️  警告: 无法访问数据加载器，跳过数据验证")
    
    # 开始训练
    print("=" * 60)
    print("开始训练...")
    print("=" * 60)
    trainer.run()


def single_node_runner(cfg, main_port: int):
    assert cfg.launcher.num_nodes == 1
    num_proc = cfg.launcher.gpus_per_node
    torch.multiprocessing.set_start_method("spawn")
    
    if num_proc == 1:
        single_proc_run(local_rank=0, main_port=main_port, cfg=cfg, world_size=num_proc)
    else:
        mp_runner = torch.multiprocessing.start_processes
        args = (main_port, cfg, num_proc)
        mp_runner(single_proc_run, args=args, nprocs=num_proc, start_method="spawn")


def format_exception(e: Exception, limit=20):
    traceback_str = "".join(traceback.format_tb(e.__traceback__, limit=limit))
    return f"{type(e).__name__}: {e}\nTraceback:\n{traceback_str}"


class SubmititRunner(submitit.helpers.Checkpointable):
    """A callable which is passed to submitit to launch the jobs."""

    def __init__(self, port, cfg):
        self.cfg = cfg
        self.port = port
        self.has_setup = False

    def run_trainer(self):
        job_env = submitit.JobEnvironment()
        add_pythonpath_to_sys_path()
        os.environ["MASTER_ADDR"] = job_env.hostnames[0]
        os.environ["MASTER_PORT"] = str(self.port)
        os.environ["RANK"] = str(job_env.global_rank)
        os.environ["LOCAL_RANK"] = str(job_env.local_rank)
        os.environ["WORLD_SIZE"] = str(job_env.num_tasks)

        # 设置数值稳定性
        setup_numerical_stability()
        
        register_omegaconf_resolvers()
        cfg_resolved = OmegaConf.to_container(self.cfg, resolve=False)
        cfg_resolved = OmegaConf.create(cfg_resolved)

        trainer = instantiate(cfg_resolved.trainer, _recursive_=False)
        
        # 检查模型参数
        if hasattr(trainer, 'model'):
            if not check_model_parameters(trainer.model):
                print("❌ 模型参数检查失败，退出训练")
                return
        
        # 验证数据加载器
        if hasattr(trainer, 'data_loader') and trainer.data_loader is not None:
            if not validate_data_loader(trainer.data_loader):
                print("❌ 数据验证失败，退出训练")
                return
        else:
            print("⚠️  警告: 无法访问数据加载器，跳过数据验证")
        
        print("=" * 60)
        print("开始训练...")
        print("=" * 60)
        trainer.run()

    def __call__(self):
        job_env = submitit.JobEnvironment()
        self.setup_job_info(job_env.job_id, job_env.global_rank)
        try:
            self.run_trainer()
        except Exception as e:
            message = format_exception(e)
            logging.error(message)
            
            # 特别处理NaN相关错误
            if "nan" in str(e).lower() or "NaN" in str(e):
                logging.error("检测到NaN损失，建议检查：")
                logging.error("1. 学习率是否过高")
                logging.error("2. 数据预处理是否正确")
                logging.error("3. 模型初始化是否正常")
                logging.error("4. 梯度裁剪设置是否合适")
            
            raise e

    def setup_job_info(self, job_id, rank):
        self.job_info = {
            "job_id": job_id,
            "rank": rank,
            "cluster": self.cfg.get("cluster", None),
            "experiment_log_dir": self.cfg.launcher.experiment_log_dir,
        }
        self.has_setup = True


def add_pythonpath_to_sys_path():
    if "PYTHONPATH" not in os.environ or not os.environ["PYTHONPATH"]:
        return
    sys.path = os.environ["PYTHONPATH"].split(":") + sys.path


def validate_paths(cfg):
    """验证所有必要的路径是否存在"""
    print("=" * 60)
    print("验证路径配置...")
    print("=" * 60)
    
    try:
        paths = cfg.paths
        required_paths = [
            ('base_annotation_path', '基础标注路径'),
            ('metaclip_img_path', 'Metaclip图像路径'),
            ('sa1b_img_path', 'SA1B图像路径'),
            ('bpe_path', 'BPE词汇路径')
        ]
        
        missing_paths = []
        for path_key, path_desc in required_paths:
            path_value = paths.get(path_key)
            if path_value and not g_pathmgr.exists(path_value):
                missing_paths.append(f"{path_desc}: {path_value}")
        
        if missing_paths:
            print("❌ 以下路径不存在:")
            for missing in missing_paths:
                print(f"  - {missing}")
            return False
        
        # 检查实验目录权限
        exp_dir = cfg.launcher.experiment_log_dir
        try:
            test_file = os.path.join(exp_dir, "test_write.txt")
            with g_pathmgr.open(test_file, "w") as f:
                f.write("test")
            g_pathmgr.rm(test_file)
        except Exception as e:
            print(f"❌ 实验目录无写权限: {exp_dir}, 错误: {e}")
            return False
        
        print("✅ 所有路径验证通过!")
        return True
        
    except Exception as e:
        print(f"❌ 路径验证失败: {e}")
        return False


def main(args) -> None:
    cfg = compose(config_name=args.config)
    if cfg.launcher.experiment_log_dir is None:
        cfg.launcher.experiment_log_dir = os.path.join(
            os.getcwd(), "sam3_logs", args.config
        )
    
    print("###################### Train App Config ####################")
    print(OmegaConf.to_yaml(cfg))
    print("############################################################")

    add_pythonpath_to_sys_path()
    makedir(cfg.launcher.experiment_log_dir)
    
    # 验证配置
    if not validate_config(cfg):
        print("❌ 配置验证失败，退出")
        return
    
    # 验证路径
    if not validate_paths(cfg):
        print("❌ 路径验证失败，退出")
        return
    
    with g_pathmgr.open(
        os.path.join(cfg.launcher.experiment_log_dir, "config.yaml"), "w"
    ) as f:
        f.write(OmegaConf.to_yaml(cfg))

    cfg_resolved = OmegaConf.to_container(cfg, resolve=False)
    cfg_resolved = OmegaConf.create(cfg_resolved)

    with g_pathmgr.open(
        os.path.join(cfg.launcher.experiment_log_dir, "config_resolved.yaml"), "w"
    ) as f:
        f.write(OmegaConf.to_yaml(cfg_resolved, resolve=True))

    submitit_conf = cfg.get("submitit", None)
    assert submitit_conf is not None, "Missing submitit config"

    experiment_log_dir = cfg.launcher.experiment_log_dir
    print(f"Experiment Log Dir:\n{experiment_log_dir}")
    submitit_dir = os.path.join(experiment_log_dir, "submitit_logs")

    # Prioritize cmd line args
    cfg.launcher.gpus_per_node = (
        args.num_gpus if args.num_gpus is not None else cfg.launcher.gpus_per_node
    )
    cfg.launcher.num_nodes = (
        args.num_nodes if args.num_nodes is not None else cfg.launcher.num_nodes
    )
    submitit_conf.use_cluster = (
        args.use_cluster if args.use_cluster is not None else submitit_conf.use_cluster
    )
    
    if submitit_conf.use_cluster:
        executor = submitit.AutoExecutor(folder=submitit_dir)
        submitit_conf.partition = (
            args.partition
            if args.partition is not None
            else submitit_conf.get("partition", None)
        )
        submitit_conf.account = (
            args.account
            if args.account is not None
            else submitit_conf.get("account", None)
        )
        submitit_conf.qos = (
            args.qos if args.qos is not None else submitit_conf.get("qos", None)
        )
        job_kwargs = {
            "timeout_min": 60 * submitit_conf.timeout_hour,
            "name": (
                submitit_conf.name if hasattr(submitit_conf, "name") else args.config
            ),
            "slurm_partition": submitit_conf.partition,
            "gpus_per_node": cfg.launcher.gpus_per_node,
            "tasks_per_node": cfg.launcher.gpus_per_node,  # one task per GPU
            "cpus_per_task": submitit_conf.cpus_per_task,
            "nodes": cfg.launcher.num_nodes,
            "slurm_additional_parameters": {
                "exclude": " ".join(submitit_conf.get("exclude_nodes", [])),
            },
        }
        if "include_nodes" in submitit_conf:
            assert (
                len(submitit_conf["include_nodes"]) >= cfg.launcher.num_nodes
            ), "Not enough nodes"
            job_kwargs["slurm_additional_parameters"]["nodelist"] = " ".join(
                submitit_conf["include_nodes"]
            )
        if submitit_conf.account is not None:
            job_kwargs["slurm_additional_parameters"]["account"] = submitit_conf.account
        if submitit_conf.qos is not None:
            job_kwargs["slurm_additional_parameters"]["qos"] = submitit_conf.qos

        if submitit_conf.get("mem_gb", None) is not None:
            job_kwargs["mem_gb"] = submitit_conf.mem_gb
        elif submitit_conf.get("mem", None) is not None:
            job_kwargs["slurm_mem"] = submitit_conf.mem

        if submitit_conf.get("constraints", None) is not None:
            job_kwargs["slurm_constraint"] = submitit_conf.constraints

        if submitit_conf.get("comment", None) is not None:
            job_kwargs["slurm_comment"] = submitit_conf.comment

        if submitit_conf.get("srun_args", None) is not None:
            job_kwargs["slurm_srun_args"] = []
            if submitit_conf.srun_args.get("cpu_bind", None) is not None:
                job_kwargs["slurm_srun_args"].extend(
                    ["--cpu-bind", submitit_conf.srun_args.cpu_bind]
                )

        print("###################### SLURM Config ####################")
        print(job_kwargs)
        print("##########################################")
        executor.update_parameters(**job_kwargs)

        if (
            "job_array" in submitit_conf
            and submitit_conf.job_array.get("num_tasks", -1) > 0
        ):
            num_tasks = submitit_conf.job_array.num_tasks
            job_array_config_dir = os.path.join(
                cfg.launcher.experiment_log_dir, "job_array_configs"
            )
            makedir(job_array_config_dir)

            job_indices = range(num_tasks)
            ports = random.sample(
                range(submitit_conf.port_range[0], submitit_conf.port_range[1] + 1),
                k=len(job_indices),
            )

            jobs_runners_configs = []
            with executor.batch():
                task_index = 0
                for indices, main_port in tqdm(zip(job_indices, ports)):
                    curr_cfg = deepcopy(cfg)
                    curr_cfg.submitit.job_array["task_index"] = task_index
                    curr_cfg_resolved = handle_custom_resolving(cfg)
                    runner = SubmititRunner(main_port, curr_cfg)
                    job = executor.submit(runner)
                    jobs_runners_configs.append(
                        (job, runner, curr_cfg, curr_cfg_resolved)
                    )
                    task_index += 1

            for job, runner, job_cfg, job_cfg_resolved in jobs_runners_configs:
                print("Submitit Job ID:", job.job_id)

                job_array_config_file = os.path.join(
                    job_array_config_dir, "{}.config.yaml".format(job.job_id)
                )
                with g_pathmgr.open(job_array_config_file, "w") as f:
                    f.write(OmegaConf.to_yaml(job_cfg))

                job_array_config_resolved_file = os.path.join(
                    job_array_config_dir, "{}.config_resolved.yaml".format(job.job_id)
                )
                with g_pathmgr.open(job_array_config_resolved_file, "w") as f:
                    f.write(OmegaConf.to_yaml(job_cfg_resolved, resolve=True))

                runner.setup_job_info(job.job_id, rank=0)
        else:
            main_port = random.randint(
                submitit_conf.port_range[0], submitit_conf.port_range[1]
            )
            runner = SubmititRunner(main_port, cfg)
            job = executor.submit(runner)
            print(f"Submitit Job ID: {job.job_id}")
            runner.setup_job_info(job.job_id, rank=0)

    else:
        cfg.launcher.num_nodes = 1
        main_port = random.randint(
            submitit_conf.port_range[0], submitit_conf.port_range[1]
        )
        single_node_runner(cfg, main_port)


if __name__ == "__main__":
    initialize_config_module("sam3.train", version_base="1.2")
    parser = ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        type=str,
        help="path to config file (e.g. configs/roboflow_v100_full_ft_100_images.yaml)",
    )
    parser.add_argument(
        "--use-cluster",
        type=int,
        default=None,
        help="whether to launch on a cluster, 0: run locally, 1: run on a cluster",
    )
    parser.add_argument("--partition", type=str, default=None, help="SLURM partition")
    parser.add_argument("--account", type=str, default=None, help="SLURM account")
    parser.add_argument("--qos", type=str, default=None, help="SLURM qos")
    parser.add_argument(
        "--num-gpus", type=int, default=None, help="number of GPUS per node"
    )
    parser.add_argument("--num-nodes", type=int, default=None, help="Number of nodes")
    args = parser.parse_args()
    args.use_cluster = bool(args.use_cluster) if args.use_cluster is not None else None
    register_omegaconf_resolvers()
    main(args)