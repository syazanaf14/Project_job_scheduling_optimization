import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.figure_factory as ff

# --- 1. FUNGSI LOAD DATA CSV ---
def load_data(file):
    if file is not None:
        df = pd.read_csv(file)
        # Jika kolum pertama adalah nama (J1, J2...), kita buang supaya hanya ambil angka
        if df.iloc[:, 0].dtype == object:
            df = df.iloc[:, 1:]
        df = df.apply(pd.to_numeric, errors='coerce')
        return df.dropna().values
    # Default Dummy Data (Jika tiada file): 5 Jobs, 10 Machines
    return np.random.randint(5, 20, size=(5, 10))

# --- 2. LOGIK EVOLUTIONARY STRATEGIES (ES) ---
def run_es(data, mu, sigma, generations, w_makespan, w_waiting, w_idle):
    # Dataset anda: Baris = Jobs (5), Lajur = Machines (10)
    n_jobs, n_machines = data.shape 
    
    # Populasi dijana berdasarkan bilangan Jobs
    population = np.random.randn(mu, n_jobs)
    history = []

    for gen in range(generations):
        # Mutasi (Gaussian)
        offspring = population + sigma * np.random.randn(mu, n_jobs)
        combined_pop = np.vstack((population, offspring))
        
        scores = []
        for ind in combined_pop:
            sequence = np.argsort(ind) # Menentukan urutan 5 Job
            
            # Matriks masa tamat (Baris: Machine, Lajur: Urutan Job)
            finish_times = np.zeros((n_machines, n_jobs))
            total_waiting_time = 0
            total_idle_time = 0
            
            for m in range(n_machines):
                for j in range(n_jobs):
                    job_idx = sequence[j]
                    
                    # Ambil p_time: Baris (Job), Lajur (Machine)
                    p_time = data[job_idx, m] 
                    
                    # Logik Pengiraan Masa (Flow Shop)
                    if m == 0 and j == 0:
                        start = 0
                    elif m == 0:
                        start = finish_times[m, j-1]
                    elif j == 0:
                        start = finish_times[m-1, j]
                    else:
                        start = max(finish_times[m-1, j], finish_times[m, j-1])
                    
                    # Kira Waiting Time & Idle Time
                    if m > 0:
                        waiting = start - finish_times[m-1, j]
                        total_waiting_time += waiting
                    if j > 0:
                        idle = start - finish_times[m, j-1]
                        total_idle_time += idle
                        
                    finish_times[m, j] = start + p_time
            
            # Fitness Score (Multi-Objective)
            makespan = finish_times[-1, -1]
            fitness = (w_makespan * makespan) + (w_waiting * total_waiting_time) + (w_idle * total_idle_time)
            scores.append(fitness)

        # Pemilihan (mu + lambda)
        indices = np.argsort(scores)[:mu]
        population = combined_pop[indices]
        history.append(scores[indices[0]])

    return history, np.argsort(population[0]), scores[indices[0]], data

# --- 3. STREAMLIT UI ---
st.set_page_config(page_title="ES Multi-Objective Optimizer", layout="wide")
st.title("⚙️ Evolutionary Strategies (ES): Multi-Objective Scheduling")
st.write("Sistem ini mengoptimumkan urutan **5 Jobs** merentasi **10 Machines**.")

# Sidebar
st.sidebar.header("Algorithmic Parameters")
uploaded_file = st.sidebar.file_uploader("Upload CSV Dataset", type="csv")
mu_val = st.sidebar.slider("Population Size (mu)", 5, 100, 50)
sigma_val = st.sidebar.slider("Step Size (sigma)", 0.01, 1.0, 0.1)
gen_val = st.sidebar.slider("Generations", 10, 500, 100)

st.sidebar.header("Objective Weights")
w_m = st.sidebar.slider("Weight: Makespan", 0.0, 1.0, 0.5)
w_w = st.sidebar.slider("Weight:
