# ComfyUI Resource Manager

通过侧栏管理模型与缓存。侧栏跟随 ComfyUI 主题，不依赖 Node 2.0。

## 切换模型自动释放

默认开启“切换模型时释放旧模型”。在下一次任务开始、加载新模型前，清除当前执行分支不再需要的模型输出与派生缓存，并回收可释放的显存和内存。

- 同一模型连续运行复用缓存。
- A → B 先释放旧 A，再加载 B。
- A+B → A+C 保留仍需使用的 A，释放 B。
- 支持原生 `CheckpointLoaderSimple`、`CheckpointLoader`、`UNETLoader`、`CLIPLoader`、`DualCLIPLoader`、`VAELoader` 中直接选择的模型。
- 以加载器输入（文件名、类型、精度等）判断切换；不检测同一路径下文件内容被替换。
- 通过连线在执行时计算模型名的任务暂不进行提前清理；第三方加载器与其内部持有的资源不在自动管理范围。

在侧栏取消该开关并保存，可停用提前清理；ComfyUI 自身的内存压力回收仍然有效。修改模型下拉框本身不会中断当前任务，也不会立即卸载模型。

## 其他功能

手动卸载 GPU 模型、完整释放模型与执行缓存、后台空闲计时、资源占用监控，以及最近 20 条手动／空闲释放记录。两项空闲计时默认关闭。GPU 卸载保留 CPU 模型，完整释放会清除缓存；进程、驱动及第三方持有的资源不保证归零。切换模型的提前清理写入后端日志，侧栏最近操作目前仅记录手动与空闲请求。

## 安装

需要 Git、Rust stable 工具链，以及已经能运行 ComfyUI 的 Python 环境。已在 Linux / CPython 3.13 / ComfyUI 0.38.0 上验证。

在 ComfyUI 根目录中，激活其 Python 虚拟环境后执行：

```bash
git clone https://github.com/buxinzi2233/ComfyUI-Resource-Manager.git custom_nodes/ComfyUI-Resource-Manager
python -m pip install -e ./custom_nodes/ComfyUI-Resource-Manager
```

安装会通过 maturin 编译 Rust 扩展。请使用启动 ComfyUI 的同一个 Python 环境；仓库不包含平台相关的 `.so` / `.pyd` 文件。

继续应用下方核心补丁，然后重启 ComfyUI。左侧“资源管理”面板提供配置入口，无需添加工作流节点。默认配置见 `config.example.json`；保存设置后会自动生成本机 `config.json`。

## 核心接口

模型切换依赖新增的 ComfyUI 缓存保留策略接口，完整补丁位于 `patches/comfyui-resource-release.patch`。补丁还修复释放请求丢失唤醒，以及 Linux 分配器未归还空闲内存的问题。

补丁基于 ComfyUI `6b747c0428c343e1417219641db93a4fb7cb69ae`（0.38.0）。在 ComfyUI 根目录执行：

```bash
git apply --check custom_nodes/ComfyUI-Resource-Manager/patches/comfyui-resource-release.patch
git apply custom_nodes/ComfyUI-Resource-Manager/patches/comfyui-resource-release.patch
```

已经应用过补丁的安装不需再次执行。更新核心后，先检查补丁与当前版本是否兼容；检查失败时不要强制应用。缺少接口时，侧栏会禁用模型切换开关并提示，原有手动／空闲释放仍可使用。补丁中的 ComfyUI 代码沿用上游 GPL-3.0，许可证随附于 `patches/LICENSE.ComfyUI`。

配置保存在 `config.json`。`unload_on_model_switch` 默认为 `true`；`unload_gpu_seconds` 和 `release_models_seconds` 为整数秒或 `null`。

## 验证

在插件目录、使用 ComfyUI 的 Python 环境执行：

```bash
python -m pip install pytest
python -m pytest -q
cargo test
python tests/check_interpreter.py
```

GPU 释放测试可用 `RESOURCE_MANAGER_TEST_GPU=1 python -m pytest tests/test_resource_release.py -q` 单独运行。测试默认使用 CPU 和隔离目录，不修改日常工作流。

已通过 35 项插件测试，并用真实 Anima 权重验证 A→A→B→A、释放后再次运行以及开关保存。测试生成的日志和证据保存在本机 `dist/`，不提交到仓库。
