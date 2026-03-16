"""
Professional PDF exporter using HTML templates and Playwright.
Falls back to legacy generator when Playwright is unavailable.
"""
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from jinja2 import Environment, FileSystemLoader, select_autoescape

from ..core.settings import REPORTS_DIR, VALIDATION_ENABLED
from .pdf_report_generator import pdf_word_generator

# Import validation utilities for data integrity
try:
    from ..utils.validators import metric_validator, validate_analysis_results
    VALIDATORS_AVAILABLE = True
except ImportError:
    VALIDATORS_AVAILABLE = False

logger = logging.getLogger(__name__)

try:
    from playwright.sync_api import sync_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    logger.warning("Playwright not available. Falling back to ReportLab PDF export.")


class ProfessionalPDFExporter:
    """Render polished PDF reports using HTML templates."""

    def __init__(self):
        templates_dir = Path("templates")
        self.environment = Environment(
            loader=FileSystemLoader(str(templates_dir)),
            autoescape=select_autoescape(['html', 'xml'])
        )
        self.export_dir = REPORTS_DIR / "exports"
        self.export_dir.mkdir(parents=True, exist_ok=True)

    def build_template_context(self, session_id: str, report: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        import re
        import html
        
        def safe_get(obj, *keys, default=None):
            ref = obj
            for key in keys:
                if isinstance(ref, dict) and key in ref:
                    ref = ref[key]
                else:
                    return default
            return ref
        
        def strip_html(text: str) -> str:
            """Remove HTML tags and decode HTML entities"""
            if not isinstance(text, str):
                return str(text) if text is not None else ""
            # Remove HTML tags
            text = re.sub(r'<[^>]+>', '', text)
            # Decode HTML entities
            text = html.unescape(text)
            return text.strip()
        
        def sanitize_percentage(value: Any, default: float = 0.0) -> float:
            """Ensure percentage is in valid 0-100 range"""
            if value is None:
                return default
            try:
                num = float(value)
                # If it's a probability (0-1), convert to percentage
                if 0 <= num <= 1:
                    num = num * 100
                # Clamp to 0-100 range
                return max(0.0, min(100.0, num))
            except (ValueError, TypeError):
                return default
        
        def sanitize_number(value: Any, default: float = 0.0, min_val: float = None, max_val: float = None) -> float:
            """Validate and clamp numeric values"""
            if value is None:
                return default
            try:
                num = float(value)
                if not isinstance(num, (int, float)) or num != num:  # Check for NaN
                    return default
                if min_val is not None:
                    num = max(min_val, num)
                if max_val is not None:
                    num = min(max_val, num)
                return num
            except (ValueError, TypeError):
                return default

        def top_n(items: List[Any], n: int = 5):
            return (items or [])[:n]

        def format_summary(summary: Any) -> str:
            if isinstance(summary, dict):
                text = summary.get("text")
                if text:
                    return text.strip()
                points = summary.get("points") or []
                if points:
                    return " ".join(points[:3])
            if isinstance(summary, str):
                return summary.strip()
            return ""

        def metric_comment(score: float, thresholds: Dict[str, str], default: str) -> str:
            for limit, message in thresholds.items():
                if score >= float(limit):
                    return message
            return default

        user_info = report.get("user") or context.get("user", {}) or {}
        video_info = context.get("video", {}) or report.get("video_metadata", {}) or {}

        timestamp = report.get("timestamp")
        if timestamp:
            try:
                formatted_timestamp = datetime.fromisoformat(timestamp).strftime("%B %d, %Y • %H:%M")
            except ValueError:
                formatted_timestamp = timestamp
        else:
            formatted_timestamp = datetime.now().strftime("%B %d, %Y • %H:%M")

        strengths = report.get("strengths") or []
        improvements = report.get("improvements") or []
        summary_points = safe_get(report, "summary", "points", default=[])
        summary_text = format_summary(report.get("summary"))

        audio_metrics = report.get("audio_analytics", {})
        visual_metrics = safe_get(report, "visual_analytics", "tension_summary", default={})
        text_metrics = report.get("text_analytics", {})

        filler_count = sanitize_number(report.get("filler_word_count"), default=0, min_val=0)
        filler_ratio = sanitize_percentage(report.get("filler_word_ratio"), default=0)
        speaking_rate = sanitize_number(safe_get(report, "speaking_metrics", "speaking_rate_wpm", default=0) or report.get("speaking_rate_wpm"), default=0, min_val=0, max_val=500)
        weak_words = sanitize_number(safe_get(report, "word_analysis", "weak_words", "weak_word_count", default=0), default=0, min_val=0)
        weak_pct = safe_get(report, "word_analysis", "weak_words", "weak_word_percentage", default=None)
        if weak_pct is None and report.get("total_words"):
            total_words_val = sanitize_number(report["total_words"], default=1, min_val=1)
            weak_pct = sanitize_percentage((weak_words / total_words_val), default=0)
        else:
            weak_pct = sanitize_percentage(weak_pct, default=0)
        
        conciseness_excess = None
        fluency_score = None
        speaking_metrics = report.get("speaking_metrics") or {}
        total_words = sanitize_number(report.get("total_words") or speaking_metrics.get("total_words"), default=0, min_val=0)
        if speaking_rate:
            if 135 <= speaking_rate <= 175:
                conciseness_excess = 0.0
            else:
                conciseness_excess = round(abs(speaking_rate - 155) / 155 * 100, 1)
            fluency_score = max(0, min(100, 100 - conciseness_excess))
        
        eye_contact_pct = safe_get(report, "visual_analytics", "tension_summary", "avg_eye_contact_pct", default=None)
        eye_contact_pct = sanitize_percentage(eye_contact_pct, default=None) if eye_contact_pct is not None else None
        
        tension_percentage = safe_get(report, "visual_analytics", "tension_summary", "tension_percentage", default=None)
        tension_percentage = sanitize_percentage(tension_percentage, default=None) if tension_percentage is not None else None

        keyword_details = text_metrics.get("keyword_details", [])
        sentence_pattern_breakdown = text_metrics.get("sentence_pattern_breakdown", {})

        pronunciation_note = metric_comment(
            report.get("voice_confidence", 0),
            {
                "85": "Pronunciation is crisp and well-articulated.",
                "70": "Mostly clear articulation; polish stress on technical terms."
            },
            "Refine enunciation and stress for technical keywords."
        )

        emotion_note = "Tone remained steady throughout the session."
        if tension_percentage is not None:
            if tension_percentage > 20:
                emotion_note = f"Detected nervous tension (~{tension_percentage:.0f}%) in several moments; add breathing resets."
            elif tension_percentage > 8:
                emotion_note = f"Mild tension (~{tension_percentage:.0f}%) early on; composure improved over time."
            else:
                emotion_note = "Warm, confident tone with minimal tension detected."

        if eye_contact_pct is not None:
            eye_contact_summary = f"{eye_contact_pct:.0f}% average eye contact"
        else:
            eye_contact_summary = "Eye contact data unavailable"

        fluency_value_display = f"{fluency_score:.0f} / 100" if fluency_score is not None else "—"
        fluency_comment = (
            f"Speaking rate {speaking_rate:.0f} wpm; keep within 140–180 range."
            if speaking_rate
            else "Speaking rate not captured."
        )
        conciseness_value_display = (
            f"{conciseness_excess:.1f}% excess" if conciseness_excess is not None else "Not captured"
        )
        conciseness_comment = (
            "Trim supporting detail and lead with headline ideas."
            if conciseness_excess is not None and conciseness_excess > 5
            else (
                "Delivery stayed tight and purposeful."
                if conciseness_excess is not None
                else "Insufficient data to evaluate conciseness."
            )
        )

        performance_overview = [
            {
                "label": "Pronunciation",
                "value": f"{report.get('voice_confidence', 0):.0f} / 100",
                "comment": pronunciation_note
            },
            {
                "label": "Fluency",
                "value": fluency_value_display,
                "comment": fluency_comment
            },
            {
                "label": "Confidence",
                "value": f"{report.get('overall_score', 0):.0f} / 100",
                "comment": "Overall delivery score across voice, visual, vocabulary."
            },
            {
                "label": "Eye Contact",
                "value": eye_contact_summary if eye_contact_pct is not None else "—",
                "comment": "Maintain triangle gaze to keep attention."
            },
            {
                "label": "Filler Ratio",
                "value": f"{filler_ratio:.1f}% ({filler_count})",
                "comment": "Excellent control of fillers." if filler_ratio < 2 else "Reduce fillers with purposeful pauses."
            },
            {
                "label": "Overall Score",
                "value": f"{report.get('overall_score', 0):.0f} / 100",
                "comment": "Composite benchmark relative to top presenters."
            },
        ]

        metrics_table = [
            {
                "category": "Pronunciation",
                "value": f"{report.get('voice_confidence', 0):.0f} / 100",
                "comment": pronunciation_note
            },
            {
                "category": "Fluency",
                "value": fluency_value_display,
                "comment": (
                    f"Pace currently {speaking_rate:.0f} wpm; target 140–180 for natural flow."
                    if speaking_rate
                    else "Speaking pace unavailable."
                )
            },
            {
                "category": "Confidence",
                "value": f"{report.get('overall_score', 0):.0f} / 100",
                "comment": "Blend vocal energy with confident posture."
            },
            {
                "category": "Filler Words",
                "value": f"{filler_ratio:.1f}% ({filler_count})",
                "comment": "Keep practicing deliberate pauses." if filler_ratio >= 2 else "Outstanding clarity—keep this up."
            },
            {
                "category": "Weak Words",
                "value": f"{weak_words} ({weak_pct:.1f}% of words)" if weak_pct is not None else f"{weak_words} instances",
                "comment": "Replace casual phrases with specific verbs."
            },
            {
                "category": "Conciseness",
                "value": conciseness_value_display,
                "comment": conciseness_comment
            },
            {
                "category": "Eye Contact",
                "value": eye_contact_summary,
                "comment": "Keep tracking audience sections left-center-right."
            },
            {
                "category": "Tension Moments",
                "value": f"{report.get('tension_count', 0)}",
                "comment": "Use shoulder roll/inhale reset between sections." if report.get("tension_count", 0) else "Body remained relaxed."
            },
        ]

        transcript_raw = report.get("enhanced_transcript") or report.get("transcript") or ""
        transcript_lines = []
        if isinstance(transcript_raw, str):
            for line in transcript_raw.splitlines():
                stripped = strip_html(line.strip())  # Strip HTML tags
                if not stripped:
                    continue
                if len(transcript_lines) >= 8:
                    break
                if ":" in stripped[:5]:
                    parts = stripped.split(" ", 1)
                    if len(parts) == 2 and parts[0].count(":") == 1:
                        transcript_lines.append({"time": parts[0], "text": strip_html(parts[1])})
                        continue
                transcript_lines.append({"time": "", "text": stripped})
        elif isinstance(transcript_raw, list):
            for seg in transcript_raw[:8]:
                transcript_lines.append({
                    "time": seg.get("timestamp") or seg.get("time") or "",
                    "text": strip_html(seg.get("text") or "")  # Strip HTML tags
                })

        # Get follow-up questions from context (should be generated from actual transcript)
        follow_up_questions = top_n(context.get("quick_prompts", []), 4)
        
        # If no questions available, try to get from report data
        if not follow_up_questions:
            report_feedback = report.get("feedback", {})
            if report_feedback and isinstance(report_feedback.get("follow_up_questions"), list):
                follow_up_questions = [q.get("question", "") if isinstance(q, dict) else str(q) 
                                      for q in report_feedback["follow_up_questions"][:4] 
                                      if q and (isinstance(q, dict) and q.get("question") or isinstance(q, str) and q.strip())]
        
        # Only use generic questions as absolute last resort (when transcript is completely empty)
        if not follow_up_questions:
            transcript = report.get("transcript") or report.get("enhanced_transcript") or ""
            if not transcript or len(transcript.strip()) < 20:
                # Empty transcript - use minimal generic questions
                follow_up_questions = [
                    "What would you like to elaborate on from your presentation?",
                    "What aspect would you like to discuss further?"
                ]
            else:
                # Should have questions but don't - extract from transcript context
                words = transcript.split()[:20]
                context_phrase = " ".join(words[-10:]) if len(words) >= 10 else " ".join(words)
                follow_up_questions = [
                    f"Can you share more details about {context_phrase}?",
                    "What would you like to expand on from your presentation?"
                ]

        visual_highlights = {
            "eye_contact": eye_contact_summary,
            "tension": f"{tension_percentage:.0f}% tension" if tension_percentage is not None else "Tension data unavailable",
            "emotion": report.get("dominant_emotion", "neutral").title()
        }

        return {
            "session_id": session_id,
            "generated_at": formatted_timestamp,
            "brand": {
                "title": "Face2Phase Communication Intelligence",
                "subtitle": "Session Performance Report"
            },
            "header": {
                "participant": user_info.get("name") or user_info.get("display_name") or "Participant",
                "email": user_info.get("email"),
                "session_id": session_id,
                "datetime": formatted_timestamp
            },
            "video": {
                "thumbnail": video_info.get("thumbnail") or video_info.get("thumbnail_url"),
                "title": video_info.get("title") or video_info.get("file_name"),
                "duration": video_info.get("duration"),
                "description": video_info.get("description")
            },
            "performance": performance_overview,
            "strengths": top_n(strengths, 4),
            "improvements": top_n(improvements, 4),
            "summary_text": summary_text,
            "metrics_table": metrics_table,
            "transcript": transcript_lines,
            "notes": {
                "pronunciation": pronunciation_note,
                "emotion": emotion_note
            },
            "visual_highlights": visual_highlights,
            "follow_up_questions": follow_up_questions,
            "insights": context.get("insights", []),
            "flags": context.get("flags", []),
            "audio": {
                "filler_trend": audio_metrics.get("filler_trend", {}),
                "pause_cadence": audio_metrics.get("pause_cadence", {}),
                "opening_confidence": audio_metrics.get("opening_confidence", {}),
                "filler_count": filler_count
            },
            "visual": {
                "tension": visual_metrics,
                "emotion_timeline": [
                    {
                        "timestamp": entry.get("timestamp"),
                        "dominant_emotion": entry.get("dominant_emotion"),
                        "confidence": sanitize_percentage(entry.get("confidence"), default=0)
                    }
                    for entry in (safe_get(report, "visual_analytics", "emotion_timeline_smoothed", default=[]) or [])
                ]
            },
            "text": {
                "keyword_coverage": text_metrics.get("keyword_coverage", {}),
                "repetition_alerts": top_n(text_metrics.get("repetition_alerts", []), 5),
                "keyword_details": keyword_details,
                "sentence_pattern_breakdown": sentence_pattern_breakdown
            }
        }

    def render_html(self, context: Dict[str, Any]) -> str:
        template = self.environment.get_template("pdf/pro_report.html")
        return template.render(**context)

    def _export_with_playwright(self, html: str, output_path: Path) -> None:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page()
            page.set_content(html, wait_until="networkidle")
            page.pdf(
                path=str(output_path),
                format="A4",
                print_background=True,
                margin={"top": "25mm", "bottom": "25mm", "left": "18mm", "right": "18mm"},
            )
            browser.close()

    def generate_pdf(self, session_id: str, report: Dict[str, Any], context: Dict[str, Any]) -> Path:
        output_path = self.export_dir / f"{session_id}_professional.pdf"

        # VALIDATION: Ensure data integrity before PDF generation
        if VALIDATORS_AVAILABLE and VALIDATION_ENABLED:
            try:
                is_valid, corrected_report, errors = validate_analysis_results(report)
                if not is_valid:
                    for error in errors:
                        logger.warning(f"Pro PDF data validation issue: {error}")
                    report = corrected_report
                    logger.info(f"Pro PDF proceeding with {len(errors)} validation corrections")
            except Exception as e:
                logger.warning(f"Could not validate report data: {e}")

        if PLAYWRIGHT_AVAILABLE:
            try:
                html = self.render_html(self.build_template_context(session_id, report, context))
                self._export_with_playwright(html, output_path)
                if not output_path.exists() or output_path.stat().st_size == 0:
                    raise RuntimeError("Playwright export produced an empty file.")
                logger.info(f"Professional PDF generated at {output_path}")
                return output_path
            except Exception as exc:
                logger.error(f"Playwright PDF generation failed: {exc}")

        # Fallback to legacy generator
        legacy_path = pdf_word_generator.generate_pdf_report(session_id, report)
        logger.info("Falling back to legacy PDF generator.")

        legacy_path_obj = Path(legacy_path) if legacy_path else None
        if legacy_path_obj and legacy_path_obj.exists() and legacy_path_obj.stat().st_size > 0:
            if legacy_path_obj.resolve() != output_path.resolve():
                try:
                    shutil.copy2(legacy_path_obj, output_path)
                    logger.info("Copied legacy PDF output to exports directory: %s", output_path)
                    return output_path
                except Exception as copy_exc:
                    logger.warning(f"Could not mirror legacy PDF into exports directory: {copy_exc}")
            return legacy_path_obj

        logger.error("Legacy PDF generator failed to produce a file.")
        raise RuntimeError("PDF generation failed: no output file was created.")


pro_pdf_exporter = ProfessionalPDFExporter()

