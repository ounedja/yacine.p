"""Format and print a diagnostic report in Arabic."""

from database import DISEASES
from model_loader import CONFIDENCE_THRESHOLD, Prediction


def format_report(
    prediction: Prediction,
    diseases: dict = DISEASES,
) -> str:
    """Create a readable Arabic report for a model prediction."""
    confidence_percent = prediction.confidence * 100

    if prediction.used_dummy:
        return "\n".join(
            [
                "=" * 48,
                "تقرير تشخيص أمراض النبات",
                "=" * 48,
                "تحذير: لم يُجرَ تشخيص؛ ملف النموذج غير متوفر.",
                f"درجة الثقة: {confidence_percent:.1f}%",
                f"حد الثقة المطلوب: {CONFIDENCE_THRESHOLD * 100:.0f}%",
                "يُرجى تثبيت نموذج مدرّب وفحص النبات لدى مختص زراعي.",
                "=" * 48,
            ]
        )

    disease = diseases.get(prediction.class_index)

    if disease is None:
        return (
            "تقرير تشخيص النبات\n"
            "تحذير: تعذّر مطابقة نتيجة النموذج مع مرض معروف."
        )

    is_confident = prediction.confidence >= CONFIDENCE_THRESHOLD
    if is_confident:
        heading = "التشخيص: تم تحديد المرض"
    else:
        heading = "تحذير: التشخيص غير مؤكد"

    disease_label = "المرض المُرجّح" if is_confident else "الاحتمال الأقرب"
    lines = [
        "=" * 48,
        "تقرير تشخيص أمراض النبات",
        "=" * 48,
        heading,
        f"{disease_label}: {disease['name_ar']} ({disease['name_en']})",
        f"نوع المسبب: {disease['pathogen_type_ar']}",
        f"درجة الثقة: {confidence_percent:.1f}%",
        f"حد الثقة المطلوب: {CONFIDENCE_THRESHOLD * 100:.0f}%",
    ]

    lines.extend(
        [
            "",
            "التوصيات الزراعية (تُراجع بعد تأكيد التشخيص):",
            disease["treatment_ar"],
            "",
            "يُرجى تأكيد التشخيص ميدانياً أو بمراجعة مختص زراعي.",
            "=" * 48,
        ]
    )
    return "\n".join(lines)


def print_report(prediction: Prediction) -> None:
    """Print the diagnostic report to the console."""
    print(format_report(prediction))