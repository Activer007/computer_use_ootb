<h2 align="center">
    <a href="https://computer-use-ootb.github.io">
        <img src="../assets/ootb_logo.png" alt="Logo" style="display: block; margin: 0 auto; filter: invert(1) brightness(2);">
    </a>
</h2>


<h5 align="center"> 如果你喜欢我们的项目，请在GitHub上为我们加星⭐以获取最新更新。</h5>

<h5 align=center>

[![arXiv](https://img.shields.io/badge/Arxiv-2411.10323-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2411.10323)
[![Project Page](https://img.shields.io/badge/Project_Page-GUI_Agent-blue)](https://computer-use-ootb.github.io)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fshowlab%2Fcomputer_use_ootb&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fshowlab%2Fcomputer_use_ootb&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)


</h5>

## <img src="../assets/ootb_icon.png" alt="Star" style="height:25px; vertical-align:middle; filter: invert(1) brightness(2);">  概览
**Computer Use <span style="color:rgb(106, 158, 210)">O</span><span style="color:rgb(111, 163, 82)">O</span><span style="color:rgb(209, 100, 94)">T</span><span style="color:rgb(238, 171, 106)">B</span>**<img src="../assets/ootb_icon.png" alt="Star" style="height:20px; vertical-align:middle; filter: invert(1) brightness(2);"> 是一个桌面GUI Agent的开箱即用（OOTB）解决方案，包括API支持的 (**Claude 3.5 Computer Use**) 和本地运行的模型 (**<span style="color:rgb(106, 158, 210)">S</span><span style="color:rgb(111, 163, 82)">h</span><span style="color:rgb(209, 100, 94)">o</span><span style="color:rgb(238, 171, 106)">w</span>UI**)。

**无需Docker**，支持 **Windows** 和 **macOS**。本项目提供了一个基于Gradio的用户友好界面。🎨

想了解更多信息，请访问我们关于Claude 3.5 Computer Use的研究 [[项目页面]](https://computer-use-ootb.github.io)。🌐

## 更新
- **<span style="color:rgb(231, 183, 98)">重大更新！</span> [2024/12/04]** **本地运行🔥** 已上线！欢迎使用 [**<span style="color:rgb(106, 158, 210)">S</span><span style="color:rgb(111, 163, 82)">h</span><span style="color:rgb(209, 100, 94)">o</span><span style="color:rgb(238, 171, 106)">w</span>UI**](https://github.com/showlab/ShowUI)，一个开源的2B视觉-语言-动作(VLA)模型作为GUI Agent。现在可兼容 `"gpt-4o + ShowUI" (~便宜200倍)`* 及 `"Qwen2-VL + ShowUI" (~便宜30倍)`*，只需几美分💰! <span style="color: grey; font-size: small;">*与Claude Computer Use相比</span>。
- **[2024/11/20]** 我们添加了一些示例来帮助你上手Claude 3.5 Computer Use。
- **[2024/11/19]** 不再受Anthropic单显示器限制——现在你可以使用 **多显示器** 🎉！
- **[2024/11/18]** 我们发布了Claude 3.5 Computer Use的深度分析: [https://arxiv.org/abs/2411.10323](https://arxiv.org/abs/2411.10323)。
- **[2024/11/11]** 不再受Anthropic低分辨率显示限制——你可以使用 *任意分辨率* 同时保持 **截图token成本较低** 🎉！
- **[2024/11/11]** 现在 **Windows** 和 **macOS** 两个平台均已支持 🎉！
- **[2024/10/25]** 现在你可以通过手机设备 📱 **远程控制** 你的电脑 💻——**无需在手机上安装APP**！试试吧，玩得开心 🎉。

## 演示视频

https://github.com/user-attachments/assets/f50b7611-2350-4712-af9e-3d31e30020ee

<div style="display: flex; justify-content: space-around;">
  <a href="https://youtu.be/Ychd-t24HZw" target="_blank" style="margin-right: 10px;">
    <img src="https://img.youtube.com/vi/Ychd-t24HZw/maxresdefault.jpg" alt="Watch the video" width="48%">
  </a>
  <a href="https://youtu.be/cvgPBazxLFM" target="_blank">
    <img src="https://img.youtube.com/vi/cvgPBazxLFM/maxresdefault.jpg" alt="Watch the video" width="48%">
  </a>
</div>


## 🚀 开始使用

### 0. 前置条件
- 请通过此[链接](https://www.anaconda.com/download?utm_source=anacondadocs&utm_medium=documentation&utm_campaign=download&utm_content=topnavalldocs)安装 Miniconda。（**Python版本：≥3.11**）
- 硬件要求（可选，针对ShowUI本地运行）：
    - **Windows (支持CUDA)**: 有CUDA支持的NVIDIA GPU，GPU显存≥6GB
    - **macOS (Apple Silicon)**: M1芯片（或更新），统一RAM≥16GB


### 1. 克隆仓库 📂
打开Conda终端。（安装Miniconda后，将在开始菜单出现）
在 **Conda终端** 中运行以下命令：
```bash
git clone https://github.com/showlab/computer_use_ootb.git
cd computer_use_ootb
```

### 2. 使用交互式安装助手（推荐）🤖

运行脚本即可检查 Conda/虚拟环境、安装依赖、下载本地模型，并生成 API Key 占位文件：

```
python install_tools/setup_ootb.py
```

该助手可以：
- 检测当前是否在 Conda/venv 中运行；
- 使用当前 Python 解释器安装 `requirements.txt`；
- 可选下载 ShowUI 全精度或 AWQ 4-bit 模型，或查看 UI-TARS / 本地 Qwen 配置提示；
- 创建 `.env` 或 `api_keys.json`，方便稍后填写密钥；
- 输出下一步操作（例如 `python app.py`、默认端口 `7860`）。

脚本可重复执行，所有步骤均为可选并具有幂等性。如果你倾向于手动配置，请继续阅读 3.1–3.4 小节。

### 3. 手动配置（如未使用助手）

#### 3.1 手动安装依赖 🔧

若未使用助手，请手动安装依赖：

```
pip install -r requirements.txt
```

#### 3.2 （可选）为 **<span style="color:rgb(106, 158, 210)">S</span><span style="color:rgb(111, 163, 82)">h</span><span style="color:rgb(209, 100, 94)">o</span><span style="color:rgb(238, 171, 106)">w</span>UI** 本地运行做准备

1. 使用以下命令下载 ShowUI-2B 模型的所有文件。确保 `ShowUI-2B` 文件夹位于 `computer_use_ootb` 目录下。

    ```
    python install_tools/install_showui.py
    ```

2. （可选，仅限 CUDA）下载 ShowUI-2B 的 AWQ 4-bit 模型：

    ```
    python install_tools/install_showui-awq-4bit.py
    ```

3. 在您的机器上安装正确的 GPU 版 PyTorch（CUDA、MPS 等）。请参考 [安装指南与验证](https://pytorch.org/get-started/locally/)。

4. 获取 [GPT-4o](https://platform.openai.com/docs/quickstart) 或 [Qwen-VL](https://help.aliyun.com/zh/dashscope/developer-reference/acquisition-and-configuration-of-api-key) 的 API Key。对于中国大陆用户，可享受 Qwen API 免费试用 100 万 token：[点击查看](https://help.aliyun.com/zh/dashscope/developer-reference/tongyi-qianwen-vl-plus-api)。

#### 3.3 （可选）为 **UI-TARS** 本地运行做准备

1. 参考 [Cloud Deployment](https://github.com/bytedance/UI-TARS?tab=readme-ov-file#cloud-deployment) 或 [VLLM 部署](https://github.com/bytedance/UI-TARS?tab=readme-ov-file#local-deployment-vllm) 指南部署 UI-TARS 服务。

2. 通过脚本 `.\install_tools\test_ui-tars_server.py` 测试 UI-TARS 服务可用性。

#### 3.4 （可选）在远程服务器部署 Qwen 规划器

1. 在 SSH 服务器上克隆本项目。

2. 启动规划器桥接：

    ```
    python computer_use_demo/remote_inference.py --host 0.0.0.0 --port 8000
    ```

### 4. 启动界面 ▶️

**启动 OOTB 界面：**
```
python app.py
```

若成功启动界面，您将在终端中看到两个 URL：
```
* Running on local URL:  http://127.0.0.1:7860
* Running on public URL: https://xxxxxxxxxxxxxxxx.gradio.live (请勿与他人分享此链接，否则他们可控制您的电脑。)
```


> <u>为方便起见</u>，我们推荐在启动界面前运行以下命令，将 API 密钥设置为环境变量。这样您无需在每次运行时手动输入。  
在 Windows Powershell 中（如在 cmd 中则使用 set 命令）：
> 
```
$env:ANTHROPIC_API_KEY="sk-xxxxx" (替换为您的密钥)
$env:QWEN_API_KEY="sk-xxxxx"
$env:OPENAI_API_KEY="sk-xxxxx"
$env:GEMINI_API_KEY="sk-xxxxx"
```

> 在 macOS/Linux 中，将上述命令中的 $env:ANTHROPIC_API_KEY 替换为 export ANTHROPIC_API_KEY 即可。


### 5. 使用任意可访问网络的设备控制您的电脑
- **待控制的电脑**：安装了上述软件的那台电脑。
- **发送指令的设备**：打开网址的任意设备。

在本机浏览器中打开 http://localhost:7860/（若在本机控制）或在您的手机浏览器中打开 https://xxxxxxxxxxxxxxxxx.gradio.live（若远程控制）。

输入 Anthropic API 密钥（可通过[此页面](https://console.anthropic.com/settings/keys)获取），然后给出指令让 AI 执行任务。

<div style="display: flex; align-items: center; gap: 10px;">
  <figure style="text-align: center;">
    <img src="./assets/gradio_interface.png" alt="Desktop Interface" style="width: auto; object-fit: contain;">
  </figure>
</div>



### 依赖与权限问题排查表

| 现象 | 可能原因 | 建议解决方案 |
| --- | --- | --- |
| `pip` 提示无写权限 | 使用系统 Python、无管理员权限 | 建议先创建 Conda 环境（如 `conda create -n ootb python=3.11`），再运行 `python install_tools/setup_ootb.py`。 |
| 安装依赖出现 SSL/代理错误 | 公司或校园网络限制 PyPI | 在 `pip.ini`/`.pip/pip.conf` 中配置代理，或手动下载离线 whl 后重新执行助手。 |
| 启动 ShowUI 时找不到 CUDA 库 | 未安装对应 CUDA 版本的 PyTorch | 参考 [官方安装指南](https://pytorch.org/get-started/locally/) 重新安装合适的 GPU 版 PyTorch。 |
| 创建 `.env` 或模型目录时报 `PermissionError` | 仓库位于只读目录 | 将仓库移动到可写路径（如 `~/computer_use_ootb`），或在具有写权限的终端中重试。 |


## 🖥️ 支持的系统
- **Windows** (Claude ✅, ShowUI ✅)
- **macOS** (Claude ✅, ShowUI ✅)

## ⚠️ 风险
- **模型可能执行危险操作**：模型仍有局限性，可能生成非预期或潜在有害的输出。建议持续监督 AI 的操作。
- **成本控制**：每个任务可能花费几美元（Claude 3.5 Computer Use）。💸

## 📅 路线图
- [ ] **探索可用功能**
  - [ ] Claude API 在解决任务时似乎不稳定。我们正在调查原因：分辨率、操作类型、操作系统平台或规划机制等。欢迎提出想法或评论。
- [ ] **界面设计**
  - [x] **支持 Gradio** ✨
  - [ ] **更简单的安装流程**
  - [ ] **更多特性**... 🚀
- [ ] **平台**
  - [x] **Windows**
  - [x] **移动端**（发出指令）
  - [x] **macOS**
  - [ ] **移动端**（被控制）
- [ ] **支持更多多模态大模型（MLLMs）**
  - [x] **Claude 3.5 Sonnet** 🎵
  - [x] **GPT-4o**
  - [x] **Qwen2-VL**
  - [ ] ...
- [ ] **改进提示策略**
  - [ ] 优化提示以降低成本。💡
- [ ] **提升推理速度**
  - [ ] 支持 int8 量化。

## 加入讨论
欢迎加入讨论，与我们一同不断改进 Computer Use - OOTB 的用户体验。可通过 [**Discord 频道**](https://discord.gg/HnHng5de) 或下方微信二维码联系我们！

<div style="display: flex; flex-direction: row; justify-content: space-around;">

<img src="../assets/wechat_2.jpg" alt="gradio_interface" width="30%">
<img src="../assets/wechat.jpg" alt="gradio_interface" width="30%">

</div>

<div style="height: 30px;"></div>

<hr>
<a href="https://computer-use-ootb.github.io">
<img src="../assets/ootb_logo.png" alt="Logo" width="30%" style="display: block; margin: 0 auto; filter: invert(1) brightness(2);">
</a>