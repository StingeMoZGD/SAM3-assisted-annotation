#!/usr/bin/env python3
"""
SAM3 自动化标注工具 — 网页版
============================
与原 tkinter 版 (sam3_annotator.py) 功能一致：
  1. 选择视频文件，浏览任意帧
  2. 在帧上标注 8 个正样本点 + 8 个负样本点（每个目标）
  3. 支持多目标标注，自动分配 obj_id
  4. 保存/加载标注项目 (JSON)
  5. 加载 SAM3 模型后运行跟踪，显示每帧的检测框
  6. 导出 YOLO 格式标注文件

使用方法:
  python app.py                # 默认 0.0.0.0:9930
  python app.py --port 9930    # 指定端口
  python app.py --checkpoint /path/to/sam3.pt --videos-dir /path/to/videos

模型权重 / BPE 词表 / 视频目录的来源优先级（从高到低）：
  命令行参数 > 环境变量 SAM3_CHECKPOINT / SAM3_BPE / SAM3_VIDEOS_DIR > 默认路径
"""

# 必须在所有 import 之前设置 TMPDIR，这样 Werkzeug 等库的内部 tempfile 也生效
# （根分区空间不足时，临时文件重定向到 /home 分区）
from flask import Flask, Response, jsonify, render_template, request
import torch
import numpy as np
import cv2
from typing import Any, Dict, Optional
from pathlib import Path
import traceback
import threading
import time
import json
import argparse
import os

_WEB_DIR = os.path.dirname(os.path.abspath(__file__))
_TMP_DIR = os.path.join(_WEB_DIR, "_tmp")
os.makedirs(_TMP_DIR, exist_ok=True)
os.environ["TMPDIR"] = _TMP_DIR


# ==================== 配置 ====================

BASE_DIR = Path(__file__).resolve().parent          # 仓库根目录
PROJECTS_DIR = BASE_DIR / "projects"
EXPORTS_DIR = BASE_DIR / "exports"


def _env_path(name, default):
    """读取环境变量指定的路径（未设置或为空则用默认值）"""
    value = os.environ.get(name, "").strip()
    return Path(value).expanduser() if value else default


VIDEOS_DIR = _env_path("SAM3_VIDEOS_DIR", BASE_DIR / "videos")
DEFAULT_CHECKPOINT = str(
    _env_path("SAM3_CHECKPOINT", BASE_DIR / "model" / "sam3.pt"))
DEFAULT_BPE = str(_env_path("SAM3_BPE", BASE_DIR /
                  "model" / "bpe_simple_vocab_16e6.txt.gz"))

# YOLO 导出的默认类别映射 (obj_id -> class_id)
ID_TO_CLASS = {3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5}

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True
# 上传视频大小限制（4GB，防超大请求）
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 ** 3


def get_available_gpus():
    """获取可用 GPU 列表"""
    try:
        if not torch.cuda.is_available():
            return []
        return list(range(torch.cuda.device_count()))
    except Exception:
        return []


# ==================== 全局状态 ====================


class AppState:
    """应用全局状态（多浏览器会话共享，与 GUI 单实例语义一致）"""

    def __init__(self):
        self.lock = threading.RLock()

        # 视频
        # {"path","name","total_frames","fps","width","height"}
        self.video: Optional[Dict[str, Any]] = None

        # 标注：{str(obj_id): {"frame_idx": int, "points": [[x,y],...], "labels": [1/0,...]}}
        self.annotations: Dict[str, Any] = {}
        self.next_obj_id = 3

        # 类别映射（可编辑）：obj_id -> class_id；class_id -> 类别名
        # （切换视频时保留，加载项目时由项目覆盖）
        self.id_to_class: Dict[int, int] = dict(ID_TO_CLASS)
        self.class_names: Dict[int, str] = {}

        # 模型
        self.predictor: Optional[Any] = None
        self.model_state = "idle"  # idle / loading / loaded / error
        self.model_msg = ""

        # 跟踪
        self.track_state = "idle"  # idle / running / done / error
        self.track_msg = ""
        self.track_done = 0
        self.track_total = 0
        # {int(frame_idx): [[obj_id, x, y, w, h], ...]}  (相对坐标)
        self.track_results: Dict[int, Any] = {}
        self.session_id: Optional[str] = None

        # 导出
        self.export_state = "idle"  # idle / running / done / error
        self.export_msg = ""
        self.export_result: Optional[Dict[str, Any]] = None


STATE = AppState()
TRACK_LOCK = threading.Lock()  # 保证跟踪任务串行执行


# ==================== 帧读取 ====================

_frame_lock = threading.Lock()
_reader = {"path": None, "cap": None, "last_idx": -1}


def read_video_frame(video_path, frame_idx):
    """读取视频指定帧（缓存 VideoCapture，顺序读取时无需 seek）"""
    with _frame_lock:
        r = _reader
        if r["path"] != video_path or r["cap"] is None:
            if r["cap"] is not None:
                r["cap"].release()
            cap = cv2.VideoCapture(video_path)
            r["path"], r["cap"], r["last_idx"] = video_path, cap, -1
        cap = r["cap"]
        if not cap.isOpened():
            return None
        # 顺序帧直接 read，否则 seek
        if r["last_idx"] != frame_idx - 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
        if not ok:
            r["last_idx"] = -1
            return None
        r["last_idx"] = frame_idx
        return frame


def _open_video_info(video_path):
    """打开视频并返回元信息；失败返回 None"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    meta_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # 部分视频文件的元数据帧数大于实际可读帧数（如录制/转码中断），
    # 从元数据末尾向前探测最多 300 帧，找到最后一个可读帧以校正帧数
    total_frames = meta_total
    if meta_total > 0:
        for probe in range(meta_total - 1, max(meta_total - 301, -1), -1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, probe)
            ok, _ = cap.read()
            if ok:
                total_frames = probe + 1
                break
    info = {
        "path": str(video_path),
        "name": os.path.basename(video_path),
        "total_frames": total_frames,
        "fps": float(cap.get(cv2.CAP_PROP_FPS)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    if info["total_frames"] <= 0:
        return None
    return info


def _resolve_video_path(raw_path):
    """把用户输入解析为视频绝对路径（相对路径优先按 videos 目录/项目根解析）"""
    p = Path(raw_path)
    if p.is_file():
        return p
    for base in (VIDEOS_DIR, BASE_DIR):
        candidate = base / raw_path
        if candidate.is_file():
            return candidate
    return None


def _reset_state_for_video(info):
    """打开/上传视频后重置标注与跟踪状态（类别映射保留，加载项目时单独恢复）"""
    with STATE.lock:
        STATE.video = info
        STATE.annotations = {}
        STATE.next_obj_id = 3
        STATE.track_state = "idle"
        STATE.track_msg = ""
        STATE.track_done = 0
        STATE.track_total = 0
        STATE.track_results = {}


# ==================== 模型加载 ====================


def _load_model_worker(gpu_id):
    """在后台线程中加载 SAM3 模型"""
    try:
        if not os.path.isfile(DEFAULT_CHECKPOINT):
            raise FileNotFoundError(
                f"未找到 SAM3 权重: {DEFAULT_CHECKPOINT}\n"
                "请将 sam3.pt 放入 model/ 目录，或用 --checkpoint 参数 / "
                "SAM3_CHECKPOINT 环境变量指定路径"
            )
        from sam3.model_builder import build_sam3_video_predictor

        predictor = build_sam3_video_predictor(
            checkpoint_path=DEFAULT_CHECKPOINT,
            bpe_path=str(DEFAULT_BPE),
            gpus_to_use=[gpu_id],
        )
        with STATE.lock:
            STATE.predictor = predictor
            STATE.model_state = "loaded"
            STATE.model_msg = ""
    except Exception as e:
        traceback.print_exc()
        with STATE.lock:
            STATE.model_state = "error"
            STATE.model_msg = str(e)


# ==================== SAM3 跟踪 ====================


def _track_worker(video_path, annotations_snapshot, width, height, total_frames):
    """跟踪工作线程（在 TRACK_LOCK 内串行执行）"""
    predictor = STATE.predictor
    if predictor is None:
        with STATE.lock:
            STATE.track_state = "error"
            STATE.track_msg = "跟踪失败: SAM3 模型未加载"
        return
    session_id = None
    try:
        with STATE.lock:
            STATE.track_state = "running"
            STATE.track_msg = "正在启动 SAM3 会话..."
            STATE.track_done = 0
            STATE.track_total = total_frames
            STATE.track_results = {}

        response = predictor.handle_request(
            request=dict(type="start_session", resource_path=video_path)
        )
        session_id = response["session_id"]
        with STATE.lock:
            STATE.session_id = session_id
            STATE.track_msg = "正在添加标注 prompt..."

        # 添加所有标注的 prompt（点坐标归一化）
        for obj_id_str, annot in annotations_snapshot.items():
            obj_id = int(obj_id_str)
            points = np.array(annot["points"], dtype=np.float32)
            labels = np.array(annot["labels"], dtype=np.int32)
            points_rel = points.copy()
            points_rel[:, 0] /= width
            points_rel[:, 1] /= height

            predictor.handle_request(
                request=dict(
                    type="add_prompt",
                    session_id=session_id,
                    frame_index=int(annot["frame_idx"]),
                    points=torch.tensor(points_rel, dtype=torch.float32),
                    point_labels=torch.tensor(labels, dtype=torch.int32),
                    obj_id=obj_id,
                )
            )

        with STATE.lock:
            STATE.track_msg = "正在传播跟踪（可能需要较长时间）..."

        # 传播跟踪（输出框为归一化相对坐标）
        # 注意 1：用点 prompt 时模型走 tracker 部分传播，必须显式传 start_frame_index，
        # 否则 previous_stages_out 为空会报 "No prompts are received" 错误
        # 注意 2：partial 传播每帧都调用 run_backbone_and_detection → conv_s0/conv_s1，
        # 而 backbone 特征在 detector 的 multigpu_buffer 中被显式缓存为 bf16
        # （见 sam3_image.py 中 to(torch.bfloat16)），conv 权重为 fp32，
        # 必须在 autocast 下运行才能自动处理混合精度，否则报
        # "Input type (c10::BFloat16) and bias type (float)" 错误
        start_frame = min(int(a["frame_idx"])
                          for a in annotations_snapshot.values())

        results = {}
        count = 0
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for out in predictor.handle_stream_request(
                request=dict(
                    type="propagate_in_video",
                    session_id=session_id,
                    start_frame_index=start_frame,
                    propagation_direction="both",
                )
            ):
                frame_idx = int(out["frame_index"])
                outputs = out["outputs"]

                boxes = outputs.get("out_boxes_xywh")
                obj_ids = outputs.get("out_obj_ids")
                if boxes is not None and len(boxes) > 0:
                    if hasattr(boxes, "cpu"):
                        boxes = boxes.cpu().numpy()
                    if hasattr(obj_ids, "cpu"):
                        obj_ids = obj_ids.cpu().numpy()
                    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
                    obj_ids = np.asarray(obj_ids).reshape(-1)
                    results[frame_idx] = [
                        [int(o), float(b[0]), float(b[1]),
                         float(b[2]), float(b[3])]
                        for o, b in zip(obj_ids, boxes)
                    ]

                count += 1
                if count % 10 == 0:
                    with STATE.lock:
                        STATE.track_done = count

        with STATE.lock:
            STATE.track_results = results
            STATE.track_state = "done"
            STATE.track_done = count
            STATE.track_msg = f"跟踪完成！共处理 {count} 帧，{len(results)} 帧含检测框"

    except Exception as e:
        traceback.print_exc()
        with STATE.lock:
            STATE.track_state = "error"
            STATE.track_msg = f"跟踪失败: {e}"
    finally:
        if session_id:
            try:
                predictor.handle_request(
                    request=dict(type="close_session", session_id=session_id)
                )
            except Exception:
                pass
            with STATE.lock:
                STATE.session_id = None


# ==================== YOLO 导出 ====================


def _export_worker(output_dir, video_path, video_name, total_frames, width, height):
    """导出 YOLO 格式标注（后台线程）"""
    try:
        with STATE.lock:
            STATE.export_state = "running"
            STATE.export_msg = "正在导出 YOLO 标注..."
            STATE.export_result = None
            track_results = dict(STATE.track_results)
            annotations = json.loads(json.dumps(STATE.annotations))
            id_to_class = {int(k): int(v)
                           for k, v in STATE.id_to_class.items()}
            class_names = {int(k): str(v)
                           for k, v in STATE.class_names.items()}

        images_dir = output_dir / "images"
        labels_dir = output_dir / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        # 未显式设置类别映射的目标 ID 按序自动分配 class_id（避免导出时被跳过）
        target_ids = sorted(int(k) for k in annotations.keys())
        used_classes = set(id_to_class.values())
        next_cls = (max(used_classes) + 1) if used_classes else 0
        for obj_id in target_ids:
            if obj_id not in id_to_class:
                id_to_class[obj_id] = next_cls
                next_cls += 1

        exported_count = 0

        # 按帧号排序，顺序读取（利用 read_video_frame 的递增优化）
        for frame_idx in sorted(track_results.keys()):
            objects = track_results[frame_idx]
            if not objects:
                continue

            selected = [
                (o, (x, y, bw, bh))
                for o, x, y, bw, bh in objects
                if int(o) in target_ids
            ]
            if not selected:
                continue

            frame = read_video_frame(video_path, frame_idx)
            if frame is None:
                continue

            img_filename = f"{video_name}_frame_{frame_idx:05d}.jpg"
            cv2.imwrite(str(images_dir / img_filename), frame)

            label_filename = f"{video_name}_frame_{frame_idx:05d}.txt"
            with open(labels_dir / label_filename, "w") as f:
                for obj_id, (x, y, bw, bh) in selected:
                    x_c = x + bw / 2
                    y_c = y + bh / 2
                    class_id = id_to_class.get(int(obj_id), -1)
                    if class_id == -1:
                        continue
                    f.write(
                        f"{class_id} {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}\n"
                    )

            exported_count += 1
            if exported_count % 10 == 0:
                with STATE.lock:
                    STATE.export_msg = f"正在导出... 已完成 {exported_count} 帧"

        # 保存配置
        config = {
            "video_path": video_path,
            "video_name": video_name,
            "total_frames": total_frames,
            "img_width": width,
            "img_height": height,
            "targets": {
                str(obj_id): {
                    "frame_idx": annot["frame_idx"],
                    "class_id": id_to_class.get(int(obj_id), -1),
                    "num_pos": int(np.sum(np.array(annot["labels"]) == 1)),
                    "num_neg": int(np.sum(np.array(annot["labels"]) == 0)),
                }
                for obj_id, annot in annotations.items()
            },
            "id_to_class": {str(k): v for k, v in id_to_class.items()},
            "class_names": {str(k): v for k, v in class_names.items()},
        }
        with open(output_dir / f"{video_name}_config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        # 生成 classes.txt（每行一个类别名，行号即 class_id，未命名行用 class_N 占位）
        max_cls_id = max(id_to_class.values(), default=-1)
        if max_cls_id >= 0:
            lines = [
                class_names.get(cid) or f"class_{cid}"
                for cid in range(max_cls_id + 1)
            ]
            with open(output_dir / "classes.txt", "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")

        with STATE.lock:
            STATE.export_state = "done"
            STATE.export_msg = f"导出完成: {exported_count} 帧已导出到 {output_dir}"
            STATE.export_result = {
                "dir": str(output_dir),
                "exported_frames": exported_count,
                "n_targets": len(annotations),
            }

    except Exception as e:
        traceback.print_exc()
        with STATE.lock:
            STATE.export_state = "error"
            STATE.export_msg = f"导出失败: {e}"


def _id_to_class_with_str_keys():
    """当前 id_to_class（JSON 存储时 key 为字符串）"""
    with STATE.lock:
        return {str(k): v for k, v in STATE.id_to_class.items()}


# ==================== Flask 路由 ====================


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/state")
def api_state():
    """获取完整应用状态"""
    with STATE.lock:
        track_frames = len(STATE.track_results)
        state = {
            "video": STATE.video,
            "annotations": STATE.annotations,
            "next_obj_id": STATE.next_obj_id,
            "id_to_class": _id_to_class_with_str_keys(),
            "class_names": {str(k): v for k, v in STATE.class_names.items()},
            "model": {"state": STATE.model_state, "msg": STATE.model_msg},
            "track": {
                "state": STATE.track_state,
                "msg": STATE.track_msg,
                "done": STATE.track_done,
                "total": STATE.track_total,
                "n_frames_with_boxes": track_frames,
            },
            "export": {
                "state": STATE.export_state,
                "msg": STATE.export_msg,
                "result": STATE.export_result,
            },
            "gpus": get_available_gpus(),
        }
    return jsonify(state)


@app.route("/api/videos")
def api_videos():
    """扫描 videos 目录列出所有视频"""
    items = []
    if VIDEOS_DIR.exists():
        for p in sorted(VIDEOS_DIR.rglob("*")):
            if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
                items.append(
                    {
                        "path": str(p),
                        "rel": str(p.relative_to(VIDEOS_DIR)),
                        "name": p.name,
                    }
                )
    return jsonify({"videos": items, "videos_dir": str(VIDEOS_DIR)})


@app.route("/api/open_video", methods=["POST"])
def api_open_video():
    """打开视频：解析元信息并重置标注与跟踪结果"""
    data = request.get_json(silent=True) or {}
    raw_path = (data.get("path") or "").strip()
    if not raw_path:
        return jsonify({"ok": False, "error": "未提供视频路径"}), 400

    with STATE.lock:
        if STATE.track_state == "running":
            return jsonify({"ok": False, "error": "跟踪正在运行，无法切换视频"}), 409

    video_path = _resolve_video_path(raw_path)
    if video_path is None:
        return jsonify({"ok": False, "error": f"视频文件不存在: {raw_path}"}), 404

    info = _open_video_info(video_path)
    if info is None:
        return jsonify({"ok": False, "error": f"无法打开视频: {video_path}"}), 400

    _reset_state_for_video(info)
    return jsonify({"ok": True, "video": info})


@app.route("/api/upload_video", methods=["POST"])
def api_upload_video():
    """上传本地视频到 videos/uploads/ 并自动打开（multipart 表单，字段名 file）"""
    with STATE.lock:
        if STATE.track_state == "running":
            return jsonify({"ok": False, "error": "跟踪正在运行，无法上传视频"}), 409

    file = request.files.get("file")
    if file is None or not file.filename:
        return jsonify({"ok": False, "error": "未收到上传文件"}), 400

    filename = _safe_name(file.filename)
    ext = Path(filename).suffix.lower()
    if ext not in VIDEO_EXTS:
        return jsonify({"ok": False, "error": f"不支持的视频格式: {ext or '(无扩展名)'}"}), 400

    uploads_dir = VIDEOS_DIR / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    dest = uploads_dir / filename
    if dest.exists():
        dest = uploads_dir / f"{Path(filename).stem}_{int(time.time())}{ext}"
    try:
        file.save(str(dest))
    except Exception as e:
        return jsonify({"ok": False, "error": f"保存上传文件失败: {e}"}), 500

    info = _open_video_info(dest)
    if info is None:
        try:
            dest.unlink()
        except OSError:
            pass
        return jsonify({"ok": False, "error": "上传的文件不是有效视频"}), 400

    _reset_state_for_video(info)
    return jsonify({"ok": True, "video": info, "rel": f"uploads/{dest.name}"})


@app.route("/api/frame")
def api_frame():
    """返回指定帧的 JPEG 图像"""
    try:
        frame_idx = int(request.args.get("index", 0))
    except ValueError:
        return jsonify({"error": "无效的帧索引"}), 400

    with STATE.lock:
        video = STATE.video
    if not video:
        return jsonify({"error": "未打开视频"}), 404

    frame_idx = max(0, min(frame_idx, video["total_frames"] - 1))
    frame = read_video_frame(video["path"], frame_idx)
    if frame is None:
        return jsonify({"error": f"无法读取第 {frame_idx} 帧"}), 500

    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        return jsonify({"error": "JPEG 编码失败"}), 500

    resp = Response(buf.tobytes(), mimetype="image/jpeg")
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/api/annotations", methods=["POST"])
def api_save_annotations():
    """保存标注（前端全量提交）"""
    data = request.get_json(silent=True) or {}
    raw_annotations = data.get("annotations")
    if not isinstance(raw_annotations, dict):
        return jsonify({"ok": False, "error": "annotations 必须为字典"}), 400

    cleaned = {}
    for obj_id_str, annot in raw_annotations.items():
        try:
            obj_id = int(obj_id_str)
            frame_idx = int(annot["frame_idx"])
            points = [[float(x), float(y)] for x, y in annot["points"]]
            labels = [int(v) for v in annot["labels"]]
        except (KeyError, TypeError, ValueError):
            continue
        entry: Dict[str, Any] = {
            "frame_idx": frame_idx,
            "points": points,
            "labels": labels,
        }
        # 多边形轮廓（可选，多边形标注模式下用于回显多边形边界）
        polygon = annot.get("polygon")
        if isinstance(polygon, list) and len(polygon) >= 3:
            try:
                entry["polygon"] = [[float(x), float(y)] for x, y in polygon]
            except (TypeError, ValueError):
                pass
        cleaned[str(obj_id)] = entry

    with STATE.lock:
        STATE.annotations = cleaned
        STATE.next_obj_id = max(
            3, int(data.get("next_obj_id", STATE.next_obj_id)))
    return jsonify({"ok": True, "count": len(cleaned)})


@app.route("/api/classes", methods=["POST"])
def api_classes():
    """更新类别映射：id_to_class（obj_id->class_id）与 class_names（class_id->名称）"""
    data = request.get_json(silent=True) or {}
    raw_id_to_class = data.get("id_to_class", {})
    raw_class_names = data.get("class_names", {})
    if not isinstance(raw_id_to_class, dict) or not isinstance(raw_class_names, dict):
        return jsonify({"ok": False, "error": "id_to_class/class_names 必须为字典"}), 400

    id_to_class = {}
    for k, v in raw_id_to_class.items():
        try:
            id_to_class[int(k)] = int(v)
        except (TypeError, ValueError):
            continue
    class_names = {}
    for k, v in raw_class_names.items():
        try:
            cid = int(k)
        except (TypeError, ValueError):
            continue
        name = str(v).strip()[:64]
        if name:
            class_names[cid] = name

    with STATE.lock:
        if STATE.track_state == "running":
            return jsonify({"ok": False, "error": "跟踪正在运行，无法修改类别"}), 409
        STATE.id_to_class = id_to_class
        STATE.class_names = class_names
    return jsonify({"ok": True})


@app.route("/api/model/load", methods=["POST"])
def api_model_load():
    """异步加载 SAM3 模型"""
    data = request.get_json(silent=True) or {}
    try:
        gpu_id = int(data.get("gpu", 0))
    except (TypeError, ValueError):
        gpu_id = 0

    with STATE.lock:
        if STATE.model_state in ("loading", "loaded"):
            return jsonify({"ok": True, "state": STATE.model_state})
        STATE.model_state = "loading"
        STATE.model_msg = ""

    thread = threading.Thread(
        target=_load_model_worker, args=(gpu_id,), daemon=True)
    thread.start()
    return jsonify({"ok": True, "state": "loading"})


@app.route("/api/track/start", methods=["POST"])
def api_track_start():
    """启动跟踪（后台线程，串行执行）"""
    with STATE.lock:
        if STATE.model_state != "loaded":
            return jsonify({"ok": False, "error": "SAM3 模型未加载"}), 400
        if not STATE.annotations:
            return jsonify({"ok": False, "error": "请先标注至少一个目标"}), 400
        if not STATE.video:
            return jsonify({"ok": False, "error": "请先选择视频"}), 400
        if STATE.track_state == "running":
            return jsonify({"ok": False, "error": "跟踪任务正在运行中"}), 409

        video = dict(STATE.video)
        annotations_snapshot = json.loads(json.dumps(STATE.annotations))

    thread = threading.Thread(
        target=_track_worker,
        args=(
            video["path"],
            annotations_snapshot,
            video["width"],
            video["height"],
            video["total_frames"],
        ),
        daemon=True,
    )
    thread.start()
    return jsonify({"ok": True})


@app.route("/api/track/frame")
def api_track_frame():
    """获取指定帧的跟踪框（相对坐标 [[obj_id, x, y, w, h], ...]）"""
    try:
        frame_idx = int(request.args.get("index", 0))
    except ValueError:
        return jsonify({"error": "无效的帧索引"}), 400

    with STATE.lock:
        objects = STATE.track_results.get(frame_idx, [])
    return jsonify({"index": frame_idx, "objects": objects})


@app.route("/api/track/frame", methods=["POST"])
def api_track_frame_update():
    """更新指定帧的跟踪框（用户在画面上拖动修正，归一化相对坐标）"""
    data = request.get_json(silent=True) or {}
    try:
        frame_idx = int(data.get("index"))
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "无效的帧索引"}), 400
    raw_objects = data.get("objects")
    if not isinstance(raw_objects, list):
        return jsonify({"ok": False, "error": "objects 必须为数组"}), 400

    cleaned = []
    for item in raw_objects:
        try:
            obj_id = int(item[0])
            x, y, w, h = (float(item[1]), float(item[2]),
                          float(item[3]), float(item[4]))
        except (TypeError, ValueError, IndexError):
            continue
        # 裁剪到图像范围内（归一化相对坐标）
        x = min(max(x, 0.0), 1.0 - 0.0005)
        y = min(max(y, 0.0), 1.0 - 0.0005)
        w = min(max(w, 0.0005), 1.0 - x)
        h = min(max(h, 0.0005), 1.0 - y)
        cleaned.append(
            [obj_id, round(x, 6), round(y, 6), round(w, 6), round(h, 6)]
        )

    with STATE.lock:
        if STATE.track_state == "running":
            return jsonify({"ok": False, "error": "跟踪正在运行，无法修改跟踪框"}), 409
        if not STATE.track_results:
            return jsonify({"ok": False, "error": "暂无跟踪结果可修改"}), 400
        STATE.track_results[frame_idx] = cleaned
    return jsonify({"ok": True, "count": len(cleaned)})


# ==================== 项目保存 / 加载 ====================


def _safe_name(name):
    """清洗文件名，去掉路径分隔符等"""
    name = os.path.basename(str(name)).strip()
    for ch in ("/", "\\", ".."):
        name = name.replace(ch, "")
    return name


@app.route("/api/project/list")
def api_project_list():
    """列出已保存的项目"""
    items = []
    if PROJECTS_DIR.exists():
        for p in sorted(PROJECTS_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
            entry = {
                "name": p.stem,
                "path": str(p),
                "mtime": p.stat().st_mtime,
                "n_targets": 0,
                "video_name": "",
            }
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                entry["n_targets"] = len(data.get("annotations", {}))
                vp = data.get("video_path") or ""
                entry["video_name"] = os.path.basename(vp)
            except Exception:
                pass
            items.append(entry)
    return jsonify({"projects": items, "projects_dir": str(PROJECTS_DIR)})


@app.route("/api/project/save", methods=["POST"])
def api_project_save():
    """保存标注项目到 projects/<name>.json（与 GUI 版格式一致）"""
    data = request.get_json(silent=True) or {}
    name = _safe_name(data.get("name") or "annotation_project")
    if not name:
        name = "annotation_project"

    with STATE.lock:
        if not STATE.annotations:
            return jsonify({"ok": False, "error": "没有可保存的标注"}), 400
        video = dict(STATE.video) if STATE.video else None
        project_data = {
            "version": "1.0",
            "video_path": video["path"] if video else None,
            "video_info": (
                {
                    "total_frames": video["total_frames"],
                    "fps": video["fps"],
                    "width": video["width"],
                    "height": video["height"],
                }
                if video
                else {}
            ),
            "next_obj_id": STATE.next_obj_id,
            "annotations": STATE.annotations,
            "id_to_class": _id_to_class_with_str_keys(),
            "class_names": {str(k): v for k, v in STATE.class_names.items()},
        }

    PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
    file_path = PROJECTS_DIR / f"{name}.json"
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(project_data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        return jsonify({"ok": False, "error": f"保存失败: {e}"}), 500

    return jsonify({"ok": True, "path": str(file_path)})


@app.route("/api/project/load", methods=["POST"])
def api_project_load():
    """加载标注项目：恢复视频 + 标注"""
    data = request.get_json(silent=True) or {}
    name = _safe_name(data.get("name") or "")
    file_path = PROJECTS_DIR / f"{name}.json"
    if not file_path.is_file():
        return jsonify({"ok": False, "error": f"项目文件不存在: {file_path}"}), 404

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            project_data = json.load(f)
    except Exception as e:
        return jsonify({"ok": False, "error": f"读取项目失败: {e}"}), 500

    warning = None

    with STATE.lock:
        if STATE.track_state == "running":
            return jsonify({"ok": False, "error": "跟踪正在运行，无法加载项目"}), 409

    # 恢复视频
    video_path = project_data.get("video_path")
    video_info = None
    if video_path and os.path.exists(video_path):
        video_info = _open_video_info(video_path)
        if video_info is None:
            warning = f"视频无法打开: {video_path}"
    else:
        warning = f"视频文件不存在: {video_path}，标注已加载，请手动选择视频"

    # 恢复标注
    annotations = {}
    for obj_id_str, annot in project_data.get("annotations", {}).items():
        try:
            entry: Dict[str, Any] = {
                "frame_idx": int(annot["frame_idx"]),
                "points": [[float(x), float(y)] for x, y in annot["points"]],
                "labels": [int(v) for v in annot["labels"]],
            }
            polygon = annot.get("polygon")
            if isinstance(polygon, list) and len(polygon) >= 3:
                try:
                    entry["polygon"] = [[float(x), float(y)]
                                        for x, y in polygon]
                except (TypeError, ValueError):
                    pass
            annotations[str(int(obj_id_str))] = entry
        except (KeyError, TypeError, ValueError):
            continue

    # 恢复类别映射
    id_to_class = {}
    for k, v in (project_data.get("id_to_class") or {}).items():
        try:
            id_to_class[int(k)] = int(v)
        except (TypeError, ValueError):
            continue
    class_names = {}
    for k, v in (project_data.get("class_names") or {}).items():
        try:
            class_names[int(k)] = str(v)
        except (TypeError, ValueError):
            continue

    with STATE.lock:
        if video_info is not None:
            STATE.video = video_info
        STATE.annotations = annotations
        STATE.next_obj_id = max(3, int(project_data.get("next_obj_id", 3)))
        if id_to_class:
            STATE.id_to_class = id_to_class
        STATE.class_names = class_names
        STATE.track_state = "idle"
        STATE.track_msg = ""
        STATE.track_done = 0
        STATE.track_total = 0
        STATE.track_results = {}

    return jsonify(
        {
            "ok": True,
            "n_targets": len(annotations),
            "video": video_info,
            "warning": warning,
        }
    )


# ==================== 导出 ====================


@app.route("/api/export", methods=["POST"])
def api_export():
    """导出 YOLO 格式标注（后台线程）"""
    data = request.get_json(silent=True) or {}
    name = _safe_name(data.get("name") or "")

    with STATE.lock:
        if STATE.export_state == "running":
            return jsonify({"ok": False, "error": "导出任务正在运行中"}), 409
        if STATE.track_state == "running":
            return jsonify({"ok": False, "error": "跟踪正在运行，请等待完成后再导出"}), 409
        if not STATE.track_results:
            return jsonify({"ok": False, "error": "没有可导出的跟踪结果，请先运行跟踪"}), 400
        video = dict(STATE.video) if STATE.video else None

    if not video:
        return jsonify({"ok": False, "error": "未打开视频"}), 400

    video_name = Path(video["path"]).stem
    if not name:
        name = video_name
    output_dir = EXPORTS_DIR / name

    thread = threading.Thread(
        target=_export_worker,
        args=(
            output_dir,
            video["path"],
            video_name,
            video["total_frames"],
            video["width"],
            video["height"],
        ),
        daemon=True,
    )
    thread.start()
    return jsonify({"ok": True, "dir": str(output_dir)})


def main():
    global DEFAULT_CHECKPOINT, DEFAULT_BPE, VIDEOS_DIR

    parser = argparse.ArgumentParser(description="SAM3 自动化标注工具（网页版）")
    parser.add_argument("--host", type=str,
                        default="0.0.0.0", help="绑定 IP，默认 0.0.0.0")
    parser.add_argument("--port", type=int, default=9930, help="绑定端口，默认 9930")
    parser.add_argument("--debug", action="store_true",
                        default=False, help="启用调试模式")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="SAM3 权重路径（默认 model/sam3.pt，可用 SAM3_CHECKPOINT 环境变量）")
    parser.add_argument("--bpe", type=str, default=None,
                        help="BPE 词表路径（默认 model/bpe_simple_vocab_16e6.txt.gz，可用 SAM3_BPE 环境变量）")
    parser.add_argument("--videos-dir", type=str, default=None,
                        help="视频目录（默认 ./videos，可用 SAM3_VIDEOS_DIR 环境变量）")
    args = parser.parse_args()

    if args.checkpoint:
        DEFAULT_CHECKPOINT = str(Path(args.checkpoint).expanduser())
    if args.bpe:
        DEFAULT_BPE = str(Path(args.bpe).expanduser())
    if args.videos_dir:
        VIDEOS_DIR = Path(args.videos_dir).expanduser()

    print(f"SAM3 标注工具（网页版）: http://{args.host}:{args.port}")
    print(f"  模型权重: {DEFAULT_CHECKPOINT}")
    print(f"  BPE 词表: {DEFAULT_BPE}")
    print(f"  视频目录: {VIDEOS_DIR}")
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()
