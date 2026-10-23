from __future__ import annotations

import hashlib
import io
import json
import os
import re
from datetime import datetime
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
import pandas as pd
import requests
import streamlit as st
from PIL import Image, ImageOps

from river_monitor import ModelRuntime, process_video, summarize

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "runs" / "train" / "river_yolo" / "weights" / "best.pt"
DEMO_IMAGE = ROOT / "datasets" / "river_debris" / "images" / "val" / "000004.jpg"
OUTPUT_DIR = ROOT / "runs" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def resolve_model_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    return path if path.is_absolute() else ROOT / path


def discover_model_path() -> Path:
    configured = os.getenv("YOLO_MODEL", "").strip()
    if configured:
        path = resolve_model_path(configured)
        if path.is_file():
            return path

    candidates = list((ROOT / "runs" / "train").glob("*/weights/best.pt"))
    if candidates:
        return max(candidates, key=lambda item: item.stat().st_mtime_ns)
    return DEFAULT_MODEL


@st.cache_resource(max_entries=3, show_spinner=False)
def load_model(weights_path: str, modified_ns: int, file_size: int) -> ModelRuntime:
    del modified_ns, file_size
    return ModelRuntime(weights_path)


def image_to_bgr(data: bytes) -> np.ndarray:
    image = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def csv_bytes(rows: list[dict]) -> bytes:
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8-sig")


def deepseek_analysis(api_key: str, model: str, result: dict) -> str:
    detections = result["detections"]
    summary = summarize(detections)
    counts_by_frame: dict[int, int] = {}
    for item in detections:
        frame = int(item.get("帧序号", 1))
        counts_by_frame[frame] = counts_by_frame.get(frame, 0) + 1
    payload = {
        "分析范围": result["kind"],
        "文件名": result["source_name"],
        "处理帧数": result["frame_count"],
        "检测框总数": summary["目标框总数"],
        "各类别检测框累计数": summary["分类统计"],
        "平均置信度": summary["平均置信度"],
        "单帧最高检测框数": max(counts_by_frame.values(), default=0),
    }
    response = requests.post(
        "https://api.deepseek.com/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "temperature": 0.2,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "你是河道漂浮物巡查分析助手。只解释所给 YOLO 类别统计、数量、置信度和处理范围；"
                        "不得推断水位、水质、水量或未提供的现场信息。视频框数是跨帧累计，不是去重物体数。"
                        "提醒人工复核，不要声称已经派发告警。用中文给出简明结论和建议。"
                    ),
                },
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
        },
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def detect_image(data: bytes, filename: str, model_path: Path, confidence: float) -> dict:
    stat = model_path.stat()
    runtime = load_model(str(model_path), stat.st_mtime_ns, stat.st_size)
    annotated, detections = runtime.detect(image_to_bgr(data), confidence)
    ok, encoded = cv2.imencode(".png", annotated)
    if not ok:
        raise RuntimeError("无法生成标注图片。")
    return {
        "kind": "图片",
        "source_name": filename,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "frame_count": 1,
        "detections": [{"帧序号": 1, "时间(秒)": 0.0, **item} for item in detections],
        "annotated_image": encoded.tobytes(),
        "annotated_video": "",
    }


def detect_video(data: bytes, filename: str, model_path: Path, confidence: float, progress, progress_note) -> dict:
    suffix = Path(filename).suffix.lower()
    if suffix not in {".mp4", ".mov", ".avi", ".mkv", ".m4v"}:
        suffix = ".mp4"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    digest = hashlib.sha256(data).hexdigest()[:8]
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(filename).stem).strip("_") or "video"
    input_path = OUTPUT_DIR / f"upload_{stamp}_{digest}{suffix}"
    output_path = OUTPUT_DIR / f"annotated_{stem}_{stamp}_{digest}.mp4"
    input_path.write_bytes(data)
    try:
        stat = model_path.stat()
        runtime = load_model(str(model_path), stat.st_mtime_ns, stat.st_size)

        def update(current: int, total: int, found: int) -> None:
            if total:
                progress.progress(min(current / total, 1.0))
                progress_note.caption(f"逐帧处理中：{current:,} / {total:,} 帧，累计检测框 {found:,} 个")
            else:
                progress_note.caption(f"逐帧处理中：已读取 {current:,} 帧，累计检测框 {found:,} 个")

        detections, frame_count, fps = process_video(
            input_path, output_path, runtime, confidence, imageio_ffmpeg.get_ffmpeg_exe(), update
        )
    finally:
        input_path.unlink(missing_ok=True)
    return {
        "kind": "视频",
        "source_name": filename,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "frame_count": frame_count,
        "fps": fps,
        "detections": detections,
        "annotated_image": b"",
        "annotated_video": str(output_path),
    }


def main() -> None:
    st.set_page_config(page_title="河道漂浮物监测平台", page_icon=":material/water:", layout="wide")
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
:root { --bg:#07131a; --line:#1d3d45; --text:#e8f1ee; --muted:#8fa8a7; --cyan:#73e6c2; }
html,body,[class*="css"] { font-family:'Manrope',sans-serif; }
.stApp { background:radial-gradient(circle at 72% -20%,#173c43 0,transparent 38%),var(--bg);color:var(--text); }
[data-testid="stSidebar"] { background:#081a20;border-right:1px solid var(--line); }
.block-container { max-width:1500px;padding:1.8rem 2.4rem 3rem; }
h1,h2,h3 { letter-spacing:0; }
.hero { border-bottom:1px solid var(--line);padding-bottom:1.2rem;margin-bottom:1rem; }
.hero h1 { margin:0;font-size:2rem;line-height:1.15;font-weight:800; }
.hero p { color:var(--muted);font-size:.82rem; }
.eyebrow { color:var(--cyan);font-family:'DM Mono';font-size:.68rem;margin-bottom:.55rem; }
.section-title { font-weight:800;margin:1.25rem 0 .65rem;font-size:1rem; }
.metric { background:linear-gradient(135deg,#10272e,#0a1d24);border:1px solid var(--line);padding:1rem;min-height:100px; }
.metric .label { color:var(--muted);font-size:.72rem; }
.metric .value { font-family:'DM Mono';font-size:1.6rem;margin-top:.5rem;font-weight:500; }
.metric .delta { font-size:.72rem;margin-top:.3rem;color:var(--cyan); }
[data-testid="stMetric"] { background:#0b1e26;border:1px solid var(--line);padding:.65rem .8rem; }
.stButton button { border-radius:3px;border:1px solid #33615f;background:#123530;color:var(--cyan);font-weight:700; }
.stButton button:hover { border-color:var(--cyan);color:#07131a;background:var(--cyan); }
[data-testid="stFileUploader"] { background:#0b1e26;border:1px dashed #376a66; }
footer { visibility:hidden; }
</style>
""",
        unsafe_allow_html=True,
    )

    st.session_state.setdefault(
        "result",
        {"kind": "", "source_name": "", "created_at": "", "frame_count": 0,
         "detections": [], "annotated_image": b"", "annotated_video": ""},
    )
    st.session_state.setdefault("deepseek_text", "")
    st.session_state.setdefault("deepseek_error", "")

    with st.sidebar:
        st.markdown("<div style='font-size:1.15rem;font-weight:800'>:material/water: RIVER SENTINEL</div>", unsafe_allow_html=True)
        st.caption("河道漂浮物与垃圾识别")
        page = st.radio("工作台", ["监测总览", "图片 / 视频检测", "检测记录", "系统设置"])
        st.divider()
        configured_model = discover_model_path()
        st.markdown("**YOLO 模型**")
        if configured_model.is_file():
            st.success("训练权重可用")
            st.caption(str(configured_model))
        else:
            st.warning("未找到训练权重")
            st.caption("检查 runs/train/river_yolo/weights/best.pt")
        st.markdown("**摄像头**")
        st.info("尚未接入")

    st.markdown(
        "<div class='hero'><div class='eyebrow'>RIVER DEBRIS INSPECTION</div>"
        "<h1>河道漂浮物监测平台</h1>"
        "<p>使用项目训练得到的 YOLO 权重识别漂浮垃圾；DeepSeek 根据检测统计辅助整理结果。</p></div>",
        unsafe_allow_html=True,
    )

    if page == "监测总览":
        render_overview()
    elif page == "图片 / 视频检测":
        render_detection()
    elif page == "检测记录":
        render_records()
    else:
        render_settings()

    st.markdown(
        "<div style='border-top:1px solid #1d3d45;margin-top:2.2rem;padding-top:.8rem;color:#607877;font-size:.7rem;font-family:DM Mono'>"
        "RIVER SENTINEL / YOLO UPLOAD INFERENCE / CAMERA NOT CONNECTED</div>",
        unsafe_allow_html=True,
    )


def render_overview() -> None:
    result = st.session_state.result
    summary = summarize(result["detections"])
    exists = bool(result["kind"])
    cards = [
        ("最近任务", result["kind"] or "暂无", result["source_name"] or "上传图片或视频后显示"),
        ("处理帧数", f"{result['frame_count']:,}" if exists else "—", "视频按所有可读帧处理"),
        ("检测框", f"{summary['目标框总数']:,}" if exists else "—", "视频同一物体可能跨帧重复"),
        ("类别数", str(summary["类别数"]) if exists else "—", "训练集定义的 8 个类别"),
    ]
    for col, (label, value, detail) in zip(st.columns(4), cards):
        with col:
            st.markdown(
                f"<div class='metric'><div class='label'>{label}</div><div class='value'>{value}</div>"
                f"<div class='delta'>{detail}</div></div>", unsafe_allow_html=True
            )
    st.markdown("<div class='section-title'>最近一次识别</div>", unsafe_allow_html=True)
    if not exists:
        st.info("进入“图片 / 视频检测”，上传河道画面后运行 YOLO。")
        if DEMO_IMAGE.is_file():
            st.caption("真实验证集样例，仅用于查看训练数据；不是现场监控画面，也未叠加模拟检测框。")
            st.image(Image.open(DEMO_IMAGE), alt="训练数据集中的真实验证集图片", width="stretch")
    elif result["kind"] == "图片":
        left, right = st.columns([1.5, 1], gap="large")
        left.image(result["annotated_image"], caption=result["source_name"], alt="YOLO 图片检测标注结果", width="stretch")
        if summary["分类统计"]:
            right.dataframe(
                pd.DataFrame([{"类别": k, "检测框数": v} for k, v in summary["分类统计"].items()]),
                hide_index=True, alt="图片检测类别统计"
            )
        else:
            right.success("当前图片未检出目标。")
    else:
        video_path = Path(result["annotated_video"])
        if video_path.is_file():
            st.video(video_path, alt="YOLO 逐帧标注后的视频")
        else:
            st.warning("标注视频已不存在，请重新处理。")
    if exists and st.session_state.deepseek_text:
        st.markdown("#### DeepSeek 分析")
        st.markdown(st.session_state.deepseek_text)
        st.caption("依据 YOLO 检测统计生成，不是 DeepSeek 独立看图识别。")


def render_detection() -> None:
    st.markdown("<div class='section-title'>上传并识别</div>", unsafe_allow_html=True)
    st.caption("图片检测整张画面；视频逐帧检测全部可读取画面。摄像头暂未接入。")
    model_default = ""
    with st.form("detection_form", border=False):
        uploaded = st.file_uploader(
            "上传河道图片或视频",
            type=["jpg", "jpeg", "png", "bmp", "webp", "mp4", "mov", "avi", "mkv", "m4v"],
        )
        model_text = st.text_input("其他 YOLO 权重路径（留空自动使用最近训练的模型）", value=model_default, help="PyCharm 新训练后会自动选择 runs/train 下最新的 weights/best.pt。")
        confidence = st.slider("置信度阈值", 0.05, 0.95, 0.25, 0.05)
        run_detection = st.form_submit_button("开始 YOLO 检测", type="primary", icon=":material/search:")

    if run_detection:
        if uploaded is None:
            st.error("请先上传图片或视频。")
        else:
            model_path = resolve_model_path(model_text.strip()) if model_text.strip() else discover_model_path()
            if not model_path.is_file():
                st.error(f"没有找到模型权重：{model_path}")
            else:
                try:
                    with st.spinner("正在读取文件并运行 YOLO…"):
                        data = uploaded.getvalue()
                        if uploaded.type.startswith("image/"):
                            result = detect_image(data, uploaded.name, model_path, confidence)
                        else:
                            progress = st.progress(0.0, text="正在准备视频检测…")
                            progress_note = st.empty()
                            result = detect_video(data, uploaded.name, model_path, confidence, progress, progress_note)
                    st.session_state.result = result
                    st.session_state.deepseek_text = ""
                    st.session_state.deepseek_error = ""
                    st.success("检测处理完成。")
                except Exception as exc:
                    st.error(f"检测失败：{type(exc).__name__}: {exc}")

    result = st.session_state.result
    if result["kind"]:
        summary = summarize(result["detections"])
        st.markdown("<div class='section-title'>检测结果</div>", unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        m1.metric("处理帧数", f"{result['frame_count']:,}")
        m2.metric("检测框累计", f"{summary['目标框总数']:,}")
        m3.metric("平均置信度", f"{summary['平均置信度']:.1%}" if summary["平均置信度"] is not None else "无目标")
        st.caption(f"{result['source_name']} · {result['created_at']}")
        if result["kind"] == "图片":
            left, right = st.columns([1.5, 1], gap="large")
            left.image(result["annotated_image"], alt="带 YOLO 检测框的图片", width="stretch")
            if summary["分类统计"]:
                right.dataframe(
                    pd.DataFrame([{"类别": k, "检测框数": v} for k, v in summary["分类统计"].items()]),
                    hide_index=True, alt="图片检测类别统计"
                )
            else:
                right.success("当前图片未检出达到阈值的目标。")
            st.download_button(
                "下载标注图片", result["annotated_image"], f"annotated_{Path(result['source_name']).stem}.png",
                "image/png", icon=":material/download:"
            )
        else:
            video_path = Path(result["annotated_video"])
            if video_path.is_file():
                st.video(video_path, alt="逐帧标注后的河道视频")
                st.download_button("下载标注视频", video_path.read_bytes(), video_path.name, "video/mp4", icon=":material/download:")
            if summary["分类统计"]:
                right_df = pd.DataFrame([{"类别": k, "各帧检测框累计数": v} for k, v in summary["分类统计"].items()])
                st.dataframe(right_df, hide_index=True, alt="视频检测类别累计统计")
            else:
                st.success("视频中没有检出达到阈值的目标。")
            st.warning("视频类别统计是跨帧检测框累计值；同一漂浮物可能重复出现，当前未做目标跟踪去重。")
        if result["detections"]:
            st.dataframe(pd.DataFrame(result["detections"]), hide_index=True, alt="逐目标检测明细")
            st.download_button(
                "下载检测明细 CSV", csv_bytes(result["detections"]),
                f"detections_{datetime.now():%Y%m%d_%H%M%S}.csv", "text/csv", icon=":material/download:"
            )
        else:
            st.info("本次已完成真实模型推理，但没有符合阈值的目标。")
        if st.session_state.deepseek_text:
            st.markdown("#### DeepSeek 分析")
            st.markdown(st.session_state.deepseek_text)
    elif DEMO_IMAGE.is_file():
        st.markdown("<div class='section-title'>真实验证集样例</div>", unsafe_allow_html=True)
        st.caption("这张图来自训练集验证集；点击按钮将实际调用 best.pt 推理，不会展示模拟检测框。")
        st.image(Image.open(DEMO_IMAGE), alt="真实河道漂浮物验证图片", width="stretch")
        if st.button("用训练权重识别这张样例", icon=":material/science:"):
            model_path = discover_model_path()
            if not model_path.is_file():
                st.error(f"训练权重不存在：{model_path}")
            else:
                try:
                    result = detect_image(DEMO_IMAGE.read_bytes(), "真实验证集样例 000004.jpg", model_path, confidence)
                    st.session_state.result = result
                    st.session_state.deepseek_text = ""
                    st.rerun()
                except Exception as exc:
                    st.error(f"样例推理失败：{type(exc).__name__}: {exc}")


def render_records() -> None:
    st.markdown("<div class='section-title'>当前会话检测记录</div>", unsafe_allow_html=True)
    result = st.session_state.result
    if not result["kind"]:
        st.info("本次会话还没有运行 YOLO 检测。")
        return
    st.caption("这里只保留当前会话最近一次结果；重启应用后不保留历史。")
    summary = summarize(result["detections"])
    st.write(f"**{result['kind']}** · {result['source_name']} · {result['created_at']}")
    st.write(f"处理 {result['frame_count']:,} 帧，检测框累计 {summary['目标框总数']:,} 个。")
    if result["detections"]:
        st.dataframe(pd.DataFrame(result["detections"]), hide_index=True, alt="最近一次任务的检测事件明细")
        st.download_button("导出事件 CSV", csv_bytes(result["detections"]), "river_detection_events.csv", "text/csv")
    else:
        st.info("本次推理没有检出目标。")


def render_settings() -> None:
    st.markdown("<div class='section-title'>系统设置</div>", unsafe_allow_html=True)
    st.markdown("#### DeepSeek 分析")
    st.caption("DeepSeek 使用 YOLO 类别统计整理结论，不会独立识别上传的画面。")
    default_key = os.getenv("DEEPSEEK_API_KEY", "")
    try:
        default_key = st.secrets.get("DEEPSEEK_API_KEY", default_key)
    except (FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
        pass
    key = st.text_input("DeepSeek API Key", value=default_key, type="password", help="仅用于本次请求，不写入项目文件。")
    model = st.selectbox("分析模型", ["deepseek-chat", "deepseek-reasoner"])
    result = st.session_state.result
    if result["kind"]:
        stats = summarize(result["detections"])
        st.write(f"当前任务：{result['kind']} · {result['source_name']} · {stats['目标框总数']} 个检测框")
        if st.button("分析最近一次 YOLO 结果", type="primary", icon=":material/psychology:"):
            if not key.strip():
                st.error("请填写 DeepSeek API Key。")
            else:
                try:
                    with st.spinner("正在请求 DeepSeek…"):
                        st.session_state.deepseek_text = deepseek_analysis(key.strip(), model, result)
                    st.session_state.deepseek_error = ""
                    st.success("分析完成。")
                except requests.RequestException as exc:
                    st.session_state.deepseek_error = str(exc)
                    st.error(f"DeepSeek 请求失败：{exc}")
                except (KeyError, ValueError, TypeError) as exc:
                    st.session_state.deepseek_error = str(exc)
                    st.error(f"DeepSeek 返回内容无法解析：{exc}")
        if st.session_state.deepseek_text:
            st.markdown(st.session_state.deepseek_text)
    else:
        st.info("先上传图片或视频运行 YOLO。")
    st.divider()
    st.markdown("#### 训练权重")
    model_path = discover_model_path()
    if model_path.is_file():
        st.success("本项目训练权重已就绪")
        st.code(str(model_path))
        st.caption("默认自动选择 runs/train 下最近更新的 best.pt；PyCharm 在本项目训练后会自动切换到新权重。")
    else:
        st.warning("还没有 best.pt；请先参考 training/PYCHARM_TRAINING.md 完成训练。")
    st.markdown("#### 识别类别")
    st.write("ball、bottle、branch、grass、leaf、milk-box、plastic-bag、plastic-garbage")
    st.caption("模型只识别训练集定义的 8 类目标；不监测水位、水质或水量，也不能保证识别训练集之外的所有垃圾类型。")
