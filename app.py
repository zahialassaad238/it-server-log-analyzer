import os
import joblib
import pandas as pd
import streamlit as st

# 1. إعدادات الصفحة
st.set_page_config(
    page_title="AI IT Log Analyzer & Dashboard",
    page_icon="🖥️",
    layout="wide"
)

st.title("🖥️ نظام تحليل خوادم الحاسوب والذكاء الاصطناعي")
st.write(
    "مرحباً بك في لوحة تحكم تحليل السجلات الذكية. "
    "قم برفع ملف السجلات (Log File) لبدء المعالجة الفورية واستخراج التقارير والحلول."
)

# 2. تحميل نموذج الذكاء الاصطناعي
MODEL_PATH = "it_log_model.joblib"

@st.cache_resource
def load_ai_model(path):
    if not os.path.exists(path):
        return None
    return joblib.load(path)

model = load_ai_model(MODEL_PATH)

if model is None:
    st.error(f"⚠️ تنبيه: لم يتم العثور على ملف النموذج ({MODEL_PATH}). يرجى تأكيد توفيره في نفس المجلد.")
    st.stop()

# 3. قاعدة المعرفة للحلول المقترحة
recommendations = {
    "Disk space is full": "تحقق من مساحة التخزين، احذف الملفات المؤقتة غير الضرورية، وقم بتوسيع المساحة.",
    "Network connection failed": "تحقق من اتصال الشبكة، إعدادات عناوين IP، وقواعد جدار الحماية (Firewall).",
    "CPU usage is high": "حدد العمليات التي تستهلك المعالج (CPU) بقوة، وراقب أداء العمليات الشاذة.",
    "Memory usage is high": "راقب استهلاك الذاكرة العشوائية (RAM) وأعد تشغيل الخدمات التي تسرب الذاكرة.",
    "Backup failed": "راجع سجلات النسخ الاحتياطي، مساحة التخزين الوجهة، والاتصال بالشبكة."
}

# 4. رفع الملفات وتجهيز الواجهة
uploaded_file = st.file_uploader(
    "اختر ملف السجلات (Log File أو TXT)",
    type=["log", "txt"]
)

if uploaded_file is not None:
    try:
        content = uploaded_file.getvalue().decode("utf-8", errors="replace")
        rows = []

        for line in content.splitlines():
            line = line.strip()
            if not line or ":" not in line:
                continue
            
            parts = line.split(":", 1)
            level = parts[0].strip().upper()
            message = parts[1].strip()
            
            rows.append({
                "Level": level,
                "Message": message
            })

        df = pd.DataFrame(rows)

        if df.empty:
            st.warning("⚠️ الملف المرفوع فارغ أو لا يتبع التنسيق الصحيح (مثال: ERROR: Disk space is full).")
            st.stop()

        # التنبؤ باستخدام الذكاء الاصطناعي
        df["AI Prediction"] = model.predict(df["Message"].tolist())

        # إحصائيات سريعة
        errors = df[df["Level"] == "ERROR"]
        warnings = df[df["Level"] == "WARNING"]
        info_logs = df[df["Level"] == "INFO"]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("إجمالي السجلات", len(df))
        col2.metric("الأخطاء (Errors)", len(errors))
        col3.metric("التحذيرات (Warnings)", len(warnings))
        col4.metric("المعلومات (Info)", len(info_logs))

        st.divider()

        # حالة النظام العامة
        st.subheader("🛡️ حالة النظام العامة")
        if len(errors) >= 3:
            st.error("🚨 خطر مرتفع (High Risk): تم رصد أخطاء متكررة تهدد استقرار النظام وتتطلب تدخلاً فورياً.")
        elif len(errors) >= 1:
            st.warning("⚠️ تنبيه (Attention Required): توجد أخطاء تتطلب المراجعة.")
        else:
            st.success("✅ الحالة طبيعية (Normal): النظام يعمل بكفاءة ودون أخطاء حرجة.")

        # تحليل الأخطاء المتكررة
        st.subheader("📊 تحليل تكرار الأخطاء")
        if not errors.empty:
            error_summary = (
                errors["Message"]
                .value_counts()
                .reset_index()
            )
            error_summary.columns = ["Error Message", "Occurrences"]
            st.dataframe(error_summary, use_container_width=True)
        else:
            st.info("لا توجد أخطاء مسجلة.")

        # التوصيات والحلول المقترحة
        st.subheader("💡 التوصيات وحلول الإصلاح")
        if not errors.empty:
            for msg in errors["Message"].unique():
                st.markdown(f"**الخطأ:** `{msg}`")
                rec = recommendations.get(msg, "مراجعة تفاصيل السجل الخاص بهذا الخطأ ومعالجته يويماً.")
                st.info(f"الإجراء المقترح: {rec}")
        else:
            st.success("لا توجد إجراءات تصحيحية مطلوبة حالياً.")

        # عرض الجدول الكامل
        st.subheader("📋 الجدول الكامل للسجلات مع تصنيفات الذكاء الاصطناعي")
        st.dataframe(df, use_container_width=True)

        # زر التحميل
        st.subheader("📥 تصدير التقرير")
        csv_data = df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="تحميل تقرير التحليل (CSV)",
            data=csv_data,
            file_name="it_log_analysis_report.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"حدث خطأ أثناء معالجة الملف: {e}")