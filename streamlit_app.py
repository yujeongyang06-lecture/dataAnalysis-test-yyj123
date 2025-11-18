#######################
# Import libraries
import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px

# ML
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

#######################
# Page configuration
st.set_page_config(
    page_title="UGV Mission Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("default")


#######################
# Load data
df_reshaped = pd.read_csv("ugv_mission_dataset_220rows.csv")

#######################
# Sidebar
with st.sidebar:

    st.header("🔧 분석 옵션 설정")

    # 1. 데이터 필터링 옵션
    st.subheader("📌 데이터 필터")

    terrain = st.selectbox(
        "Terrain Type 선택",
        options=sorted(df_reshaped["TerrainType"].unique())
    )

    obstacle = st.slider(
        "Obstacle Density (장애물 밀도)", 
        min_value=int(df_reshaped["ObstacleDensity"].min()),
        max_value=int(df_reshaped["ObstacleDensity"].max()),
        value=1
    )

    success_filter = st.radio(
        "Mission Success 여부 필터",
        options=["전체", "성공(1)", "실패(0)"]
    )

    sensor_min = st.slider(
        "SensorHealth 최소값", 
        min_value=0, 
        max_value=100, 
        value=50
    )

    st.markdown("---")

    # 2. 머신러닝 모델 설정
    st.subheader("🤖 머신러닝 모델 옵션")

    model_type = st.radio(
        "예측 모델 선택",
        options=["Logistic Regression", "Random Forest"]
    )

    test_size = st.slider(
        "Train/Test 비율 설정 (Test Size)",
        min_value=0.1, 
        max_value=0.4, 
        value=0.2
    )

    st.markdown("---")

    # 3. 테마 설정
    st.subheader("🎨 시각화 테마")
    theme = st.selectbox(
        "테마 선택",
        options=["Light", "Dark"]
    )

    st.markdown("---")

    # 4. 앱 설명
    st.subheader("ℹ️ 대시보드 설명")
    st.write("""
    이 대시보드는 UGV(무인 지상 차량) 임무 데이터를 기반으로  
    **지형, 장애물, 배터리, 센서 상태 등이 임무 성공률에 미치는 영향**을 분석하고  
    머신러닝 모델로 **Mission Success 예측**을 수행합니다.
    """)


#######################
# Plots & Dashboard Layout

# 👉 여기를 넓게 변경함 (col[0]이 좁았던 문제 해결)
col = st.columns((3, 5, 3), gap='medium')
with col[0]:
    st.subheader("📌 주요 임무 지표 (KPI)")

    # KPI 계산
    avg_battery = df_reshaped["BatteryLevel"].mean()
    avg_speed = df_reshaped["Speed"].mean()
    avg_sensor = df_reshaped["SensorHealth"].mean()
    success_rate = df_reshaped["MissionSuccess"].mean() * 100

    # KPI 카드 표시
    kpi_col1, kpi_col2 = st.columns(2)
    with kpi_col1:
        st.metric("평균 배터리 (%)", f"{avg_battery:.1f}")
        st.metric("평균 속도 (km/h)", f"{avg_speed:.2f}")

    with kpi_col2:
        st.metric("평균 센서 상태 (%)", f"{avg_sensor:.1f}")
        st.metric("미션 성공률 (%)", f"{success_rate:.1f}")

    st.markdown("---")

    # -----------------------------
    # ① TerrainType별 Mission Success 시각화
    # -----------------------------
    st.subheader("🗺️ 지형(TerrainType)별 미션 성공률")

    terrain_success = (
        df_reshaped.groupby("TerrainType")["MissionSuccess"].mean().reset_index()
    )
    terrain_success["MissionSuccess"] = terrain_success["MissionSuccess"] * 100

    terrain_chart = alt.Chart(terrain_success).mark_bar().encode(
        x=alt.X("TerrainType:O", title="Terrain Type"),
        y=alt.Y("MissionSuccess:Q", title="Success Rate (%)"),
        color="TerrainType:O"
    ).properties(height=260)

    st.altair_chart(terrain_chart, use_container_width=True)

    st.markdown("---")

    # -----------------------------
    # ② 장애물 밀도별 평균 MissionTime
    # -----------------------------
    st.subheader("🚧 장애물 밀도별 평균 Mission Time")

    obstacle_time = (
        df_reshaped.groupby("ObstacleDensity")["MissionTime"]
        .mean()
        .reset_index()
    )

    obstacle_chart = alt.Chart(obstacle_time).mark_line(point=True).encode(
        x=alt.X("ObstacleDensity:O", title="Obstacle Density"),
        y=alt.Y("MissionTime:Q", title="Avg Mission Time (min)"),
        color="ObstacleDensity:O"
    ).properties(height=260)

    st.altair_chart(obstacle_chart, use_container_width=True)


with col[1]:

    st.subheader("📊 변수 간 상관관계 (Correlation Heatmap)")

    # 수치형 데이터만 추출
    numeric_df = df_reshaped.select_dtypes(include=['int64', 'float64'])
    corr = numeric_df.corr().reset_index().melt('index')

    heatmap_chart = alt.Chart(corr).mark_rect().encode(
        x=alt.X('variable:O', title=""),
        y=alt.Y('index:O', title=""),
        color=alt.Color('value:Q', scale=alt.Scale(scheme='blues')),
        tooltip=['index', 'variable', 'value']
    ).properties(height=400)

    st.altair_chart(heatmap_chart, use_container_width=True)

    st.markdown("---")

    # --------------------------
    # MissionTime 관련 Scatterplots
    # --------------------------

    st.subheader("⏱ Mission Time 영향 변수 분석")

    scatter_cols = st.columns(3)

    # 1) BatteryLevel vs MissionTime
    with scatter_cols[0]:
        st.markdown("**🔋 BatteryLevel vs MissionTime**")
        chart1 = alt.Chart(df_reshaped).mark_circle(size=60).encode(
            x="BatteryLevel",
            y="MissionTime",
            color="MissionSuccess:N",
            tooltip=["BatteryLevel", "MissionTime", "MissionSuccess"]
        ).interactive()
        st.altair_chart(chart1, use_container_width=True)

    # 2) PayloadWeight vs MissionTime
    with scatter_cols[1]:
        st.markdown("**📦 PayloadWeight vs MissionTime**")
        chart2 = alt.Chart(df_reshaped).mark_circle(size=60).encode(
            x="PayloadWeight",
            y="MissionTime",
            color="MissionSuccess:N",
            tooltip=["PayloadWeight", "MissionTime", "MissionSuccess"]
        ).interactive()
        st.altair_chart(chart2, use_container_width=True)

    # 3) Speed vs MissionTime
    with scatter_cols[2]:
        st.markdown("**🚗 Speed vs MissionTime**")
        chart3 = alt.Chart(df_reshaped).mark_circle(size=60).encode(
            x="Speed",
            y="MissionTime",
            color="MissionSuccess:N",
            tooltip=["Speed", "MissionTime", "MissionSuccess"]
        ).interactive()
        st.altair_chart(chart3, use_container_width=True)

    st.markdown("---")

    # --------------------------
    # TerrainType × ObstacleDensity → 평균 Speed Heatmap
    # --------------------------

    st.subheader("🌡 지형 × 장애물 → 평균 Speed Heatmap")

    pivot_df = (
        df_reshaped
        .groupby(["TerrainType", "ObstacleDensity"])["Speed"]
        .mean()
        .reset_index()
    )

    speed_heatmap = alt.Chart(pivot_df).mark_rect().encode(
        x=alt.X("ObstacleDensity:O", title="Obstacle Density"),
        y=alt.Y("TerrainType:O", title="Terrain Type"),
        color=alt.Color("Speed:Q", scale=alt.Scale(scheme='viridis')),
        tooltip=["TerrainType", "ObstacleDensity", "Speed"]
    ).properties(height=300)

    st.altair_chart(speed_heatmap, use_container_width=True)


with col[2]:

    st.subheader("🤖 미션 성공 예측 (Machine Learning)")

    # ----------------------------------------
    # 1) Feature / Target 분리
    # ----------------------------------------
    X = df_reshaped[[
        "TerrainType", "BatteryLevel", "PayloadWeight",
        "CommQuality", "SensorHealth", "ObstacleDensity", "Speed", "MissionTime"
    ]]
    y = df_reshaped["MissionSuccess"]

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    # ----------------------------------------
    # 2) 모델 선택 (사이드바 radio)
    # ----------------------------------------
    if model_type == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)
    else:
        model = RandomForestClassifier(n_estimators=200, random_state=42)

    # 모델 학습
    model.fit(X_train, y_train)

    # 예측
    y_pred = model.predict(X_test)

    # 정확도 표시
    from sklearn.metrics import accuracy_score, confusion_matrix
    accuracy = accuracy_score(y_test, y_pred)

    st.metric("모델 정확도 (Accuracy)", f"{accuracy * 100:.2f}%")
    st.markdown("---")

    # ----------------------------------------
    # 3) Confusion Matrix 출력
    # ----------------------------------------
    st.subheader("📟 Confusion Matrix")

    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm,
                         index=["Actual 0 (Fail)", "Actual 1 (Success)"],
                         columns=["Pred 0", "Pred 1"])

    cm_chart = px.imshow(
        cm_df,
        text_auto=True,
        color_continuous_scale="blues",
        aspect="auto",
        title="Confusion Matrix"
    )
    st.plotly_chart(cm_chart, use_container_width=True)

    st.markdown("---")

    # ----------------------------------------
    # 4) Feature Importance (RF일 때만)
    # ----------------------------------------
    if model_type == "Random Forest":
        st.subheader("🌳 Feature Importance")

        importance_df = pd.DataFrame({
            "Feature": X.columns,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=False)

        bar_chart = alt.Chart(importance_df).mark_bar().encode(
            x=alt.X("Importance:Q"),
            y=alt.Y("Feature:N", sort='-x'),
            color="Importance:Q"
        ).properties(height=300)

        st.altair_chart(bar_chart, use_container_width=True)

        st.markdown("---")

    # ----------------------------------------
    # 5) 실시간 예측 기능
    # ----------------------------------------
    st.subheader("🎯 실시간 미션 성공 예측")

    st.write("사이드바 입력을 기반으로 MissionSuccess(0/1)를 예측합니다.")

    input_data = pd.DataFrame({
        "TerrainType": [terrain],
        "BatteryLevel": [df_reshaped["BatteryLevel"].mean()],  # 사용자가 원하면 변경 가능
        "PayloadWeight": [df_reshaped["PayloadWeight"].mean()],
        "CommQuality": [df_reshaped["CommQuality"].mean()],
        "SensorHealth": [sensor_min],
        "ObstacleDensity": [obstacle],
        "Speed": [df_reshaped["Speed"].mean()],
        "MissionTime": [df_reshaped["MissionTime"].mean()]
    })

    pred_result = model.predict(input_data)[0]

    if pred_result == 1:
        st.success("🚀 **예측 결과: 미션 성공 (Success, 1)**")
    else:
        st.error("💥 **예측 결과: 미션 실패 (Fail, 0)**")
