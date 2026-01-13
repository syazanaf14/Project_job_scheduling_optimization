import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.figure_factory as ff

# --- FUNGSI LOAD DATA ---
def load_data(file):
    if file is not None:
        df = pd.read_csv(file)
        if df.iloc[:, 0].dtype == object:
            df = df.iloc[:, 1:]
        df = df.apply(pd.to_numeric, errors='coerce')
        return df.dropna().values
    return np.array([[10, 20, 5, 15], [8, 12, 20, 10], [15, 5, 10, 25]])

# --- LOGIK EVOLUTIONARY STRATEGIES (ES) DENGAN MULTI-OBJECTIVE ---
def run_es(data, mu, sigma, generations, w_makespan, w_waiting, w_idle):
    n_machines, n_jobs = data.shape
    population = np.random.randn(mu, n_jobs)
    history = []

    for gen in range(generations):
        offspring = population + sigma * np.random.randn(mu, n_jobs)
        combined_pop = np.vstack((population, offspring))
        
        scores = []
        for ind in combined_pop:
            sequence = np.argsort(ind)
            
            finish_times = np.zeros((n_machines, n_jobs))
            total_waiting_time = 0
            total_idle_time = 0
            
            for m in range(n_machines):
                for j in range(n_jobs):
                    job_idx = sequence[j]
                    p_time = data[m, job_idx]
                    
                    # Kira Start Time
                    if m == 0 and j == 0:
                        start = 0
                    elif m == 0:
                        start = finish_times[m, j-1]
                    elif j == 0:
                        start = finish_times[m-1, j]
                    else:
                        start = max(finish_times[m-1, j], finish_times[m, j-1])
                    
                    # --- KIRA WAITING TIME (Masa Job menunggu mesin) ---
                    if m > 0:
                        waiting = start - finish_times[m-1, j]
                        total_waiting_time += waiting
                        
                    # --- KIRA MACHINE IDLE TIME (Masa Mesin menunggu job) ---
                    if j > 0:
                        idle = start - finish_times[m, j-1]
                        total_idle_time += idle
                        
                    finish_times[m, j] = start + p_time
            
            makespan = finish_times[-1, -1]
            
            # --- FORMULA FITNESS (Weighted Sum) ---
            # Minimumkan nilai ini
            fitness = (w_makespan * makespan) + (w_waiting * total_waiting_time) + (w_idle * total_idle_time)
            scores.append(fitness)

        indices = np.argsort(scores)[:mu]
        population = combined_pop[indices]
        history.append(scores[indices[0]])

    # Return data tambahan untuk paparan metrik
    return history, np.argsort(population[0]), scores[indices[0]], data

# --- STREAMLIT UI ---
st.set_page_config(page_title="ES Multi-Objective Optimizer", layout="wide")
st.title("⚙️ Evolutionary Strategies (ES): Multi-Objective Scheduling")
st.write("Sistem ini mengoptimumkan jadual berdasarkan Makespan, Waiting Time, dan Idle Time.")

# Sidebar Parameter 
st.sidebar.header("Algorithmic Parameters")
uploaded_file = st.sidebar.file_uploader("Upload CSV Dataset", type="csv")
mu_val = st.sidebar.slider("Population Size (mu)", 5, 100, 20)
sigma_val = st.sidebar.slider("Step Size (sigma)", 0.01, 1.0, 0.1)
gen_val = st.sidebar.slider("Generations", 10, 500, 100)

# --- TAMBAHAN: TOGGLE WEIGHTS UNTUK MULTI-OBJECTIVE ---
st.sidebar.header("Objective Weights")
st.sidebar.info("Tentukan kepentingan setiap objektif (Jumlah = 1.0 disarankan)")
w_m = st.sidebar.slider("Weight: Makespan", 0.0, 1.0, 0.7)
w_w = st.sidebar.slider("Weight: Job Waiting Time", 0.0, 1.0, 0.2)
w_i = st.sidebar.slider("Weight: Machine Idle Time", 0.0, 1.0, 0.1)

if st.button("Start Multi-Objective For ES Optimization"):
    data = load_data(uploaded_file)
    # call fungsi ES dengan parameter weights
    hist, best_seq, best_fitness, raw_data = run_es(data, mu_val, sigma_val, gen_val, w_m, w_w, w_i)

    # Kira semula komponen untuk paparan metrik akhir
    n_m, n_j = raw_data.shape
    f_times = np.zeros((n_m, n_j))
    t_wait = 0
    t_idle = 0
    for m in range(n_m):
        for j in range(n_j):
            idx = best_seq[j]
            st_time = 0 if (m==0 and j==0) else (f_times[m,j-1] if m==0 else (f_times[m-1,j] if j==0 else max(f_times[m-1,j], f_times[m,j-1])))
            if m > 0: t_wait += (st_time - f_times[m-1, j])
            if j > 0: t_idle += (st_time - f_times[m, j-1])
            f_times[m, j] = st_time + raw_data[m, idx]

    # 1. Metrics Display
    c1, c2, c3 = st.columns(3)
    c1.metric("Optimized Makespan", f"{f_times[-1,-1]} mins")
    c2.metric("Total Waiting Time", f"{t_wait} mins")
    c3.metric("Total Machine Idle Time", f"{t_idle} mins")
    
    st.success(f"**Best Sequence Found:** {best_seq}")

    # 2. Convergence Plot Graph
    st.subheader("📈 Convergence Analysis (Weighted Fitness)")
    fig, ax = plt.subplots()
    ax.plot(hist, label='Best Fitness Score', color='green')
    ax.set_xlabel("Generation")
    ax.set_ylabel("Combined Fitness Value")
    ax.legend()
    st.pyplot(fig)

    # 3. Gantt Chart for the job schedule
    st.subheader("📅 Optimized Gantt Chart")
    gantt_data = []
    for m in range(n_m):
        for j in range(n_j):
            idx = best_seq[j]
            p_time = raw_data[m, idx]
            st_t = 0 if (m==0 and j==0) else (f_times[m,j-1] if m==0 else (f_times[m-1,j] if j==0 else max(f_times[m-1,j], f_times[m,j-1])))
            en_t = st_t + p_time
            gantt_data.append(dict(Task=f"Machine {m+1}", Start=st_t, Finish=en_t, Resource=f"Job {idx+1}"))

    df_plot = pd.DataFrame(gantt_data)
    df_plot['Start'] = pd.to_datetime(df_plot['Start'], unit='m', origin='2026-01-01')
    df_plot['Finish'] = pd.to_datetime(df_plot['Finish'], unit='m', origin='2026-01-01')
    fig_gantt = ff.create_gantt(df_plot, index_col='Resource', show_colorbar=True, group_tasks=True, showgrid_x=True)
    st.plotly_chart(fig_gantt, use_container_width=True)
