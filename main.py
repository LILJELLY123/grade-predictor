import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from fastapi import FastAPI
import uvicorn
import streamlit as st
import pickle
from pathlib import Path

# 1. Генерация данных и обучение модели
def generate_and_train():
    np.random.seed(42)

    # параметры студентов
    data = {
        'study_hours': np.clip(np.random.normal(loc=2, scale=1, size=500), 0.5, 8),  # Диапазон до 8 часов
        'days_before_exam': np.random.randint(1, 31, 500),
        'previous_grades': np.clip(np.random.normal(loc=3.5, scale=0.8, size=500), 2, 5),
    }

    # Формула расчета оценки
    data['grade'] = (np.clip(
        data['previous_grades'] * 0.6 +
        np.log1p(data['study_hours'] * data['days_before_exam']) * 0.4 +
        np.random.normal(0, 0.3, 500),
        2, 5  # Оценки от 2 до 5
    ))

    # Округление до 0.5 (чтобы были оценки типа 3.5)
    data['grade'] = np.round(data['grade'] * 2) / 2

    df = pd.DataFrame(data)

    model = RandomForestRegressor(n_estimators=150, random_state=42)
    model.fit(df[['study_hours', 'days_before_exam', 'previous_grades']], df['grade'])

    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)

    return model


# 2. FastAPI сервер
app = FastAPI()
model = generate_and_train()


@app.post("/predict")
def predict(study_hours: float, days: int, previous_grade: float = 3.5):
    grade = model.predict([[study_hours, days, previous_grade]])[0]
    # Округляем до ближайшего 0.5
    final_grade = np.clip(round(grade * 2) / 2, 2, 5)
    return {"grade": float(final_grade)}


def run_api():
    uvicorn.run(app, host="0.0.0.0", port=8000)


# 3. Streamlit интерфейс
def run_streamlit():
    # Загрузка модели
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        st.success("Модель успешно загружена!")
    except FileNotFoundError:
        st.error("Модель не найдена. Пожалуйста, убедитесь, что файл model.pkl существует.")
        return

    st.title("🎓 Предсказатель оценок ")

    st.markdown("""
    <style>
    .big-font { font-size:18px !important; }
    </style>
    """, unsafe_allow_html=True)

    # Загрузка датасета с данными студентов
    uploaded_file = st.file_uploader("Загрузите CSV-файл с данными студентов", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # Проверка необходимых колонок
        required_columns = ['study_hours', 'days_before_exam', 'previous_grades']
        if all(col in df.columns for col in required_columns):
            st.write("Загруженный датасет:")
            st.write(df)

            # Предсказание оценок
            predictions = model.predict(df[required_columns])
            df['predicted_grade'] = np.clip(np.round(predictions * 2) / 2, 2, 5)

            # Вывод результатов
            st.write("Результаты предсказания:")
            st.write(df)

            # Предложение скачать результаты
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Скачать результаты в CSV",
                data=csv,
                file_name="predicted_grades.csv",
                mime="text/csv",
            )
        else:
            st.error(f"Загруженный датасет должен содержать следующие колонки: {required_columns}")

    # Ручной ввод данных
    st.subheader("Ручной ввод данных")
    col1, col2 = st.columns(2)
    with col1:
        study_hours = st.slider("Часов в день", 0.5, 8.0, 2.0, step=0.5,  # Диапазон до 8 часов
                                help="Сколько часов в день вы готовитесь")
    with col2:
        days = st.number_input("Дней до экзамена", 1, 30, 14,
                               help="Сколько дней осталось на подготовку")

    previous_grade = st.select_slider("Текущий уровень",
                                      options=[2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
                                      value=3.5)

    if st.button("Предсказать оценку", type="primary"):
        try:
            grade = model.predict([[study_hours, days, previous_grade]])[0]
            final_grade = np.clip(round(grade * 2) / 2, 2, 5)

            # Отображение результата
            st.subheader(f"Прогнозируемая оценка: {final_grade}")

            # Визуальная обратная связь
            if final_grade >= 4.5:
                st.success("Отлично! Вы хорошо подготовлены")
                st.balloons()
            elif final_grade >= 3.5:
                st.info("Хорошо, но можно лучше")
            else:
                st.warning("Риск неудовлетворительной оценки")
                st.error("Рекомендуем увеличить время подготовки")

            # Советы
            if study_hours < 1.5 and final_grade < 4.0:
                st.markdown("🔹 **Совет:** Попробуйте заниматься хотя бы 2 часа в день")
            if days < 7 and final_grade < 4.0:
                st.markdown("🔹 **Совет:** У вас мало времени - увеличьте интенсивность подготовки")

        except Exception as e:
            st.error(f"Ошибка: {str(e)}")


# --- Запуск ---
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "api":
        run_api()
    else:
        run_streamlit()
