import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

sns.set(style='dark')

def create_monthly_orders_df(df):
    df['order_month'] = df['order_purchase_timestamp'].dt.to_period('M').astype(str)
    monthly_orders = df.groupby('order_month').size().reset_index(name='order_count')
    
    return monthly_orders

def create_category_sales(df):
    category_sales = df.groupby('product_category_name_english').agg(
        total_units=('order_item_id', 'count'),
        total_revenue=('price', 'sum')
    ).reset_index()

    return category_sales

def create_payment_methods_df(df):
    payment_methods = df.groupby('payment_type').agg(
        frequency=('order_id', 'count'),
        total_value=('payment_value', 'sum')
    ).reset_index()
    payment_methods['percentage'] = payment_methods['frequency'] / payment_methods['frequency'].sum() * 100
    payment_methods = payment_methods.sort_values('frequency', ascending=False)
    
    return payment_methods

def create_review_counts_df(df):
    review_counts = df['review_score'].value_counts().sort_index().reset_index()
    review_counts.columns = ['score', 'count']
    review_counts['percentage'] = review_counts['count'] / review_counts['count'].sum() * 100

    return review_counts

def create_top_cities_df(df):
    # Persiapan data
    location_purchases = df.groupby(['customer_city', 'customer_state']).size().reset_index(name='order_count')
    top_cities = location_purchases.sort_values('order_count', ascending=False).head(5)

    return top_cities

def create_top_states_df(df):
    state_purchases = df.groupby('customer_state').size().reset_index(name='order_count')
    state_purchases = state_purchases.sort_values('order_count', ascending=False).head(5)

    return state_purchases

def create_rfm_df(df):
    rfm_df = df.groupby(by="customer_id_x", as_index=False).agg({
        "order_purchase_timestamp": "max", # mengambil tanggal order terakhir
        "order_id": "nunique", # menghitung jumlah order
        "price": "sum" # menghitung jumlah revenue yang dihasilkan
    })
    rfm_df.columns = ["customer_id", "max_order_timestamp", "frequency", "monetary"]

    # menghitung kapan terakhir pelanggan melakukan transaksi (hari)
    rfm_df["max_order_timestamp"] = rfm_df["max_order_timestamp"].dt.date
    recent_date = df["order_purchase_timestamp"].dt.date.max()
    rfm_df["recency"] = rfm_df["max_order_timestamp"].apply(lambda x: (recent_date - x).days)

    rfm_df.drop("max_order_timestamp", axis=1, inplace=True)

    rfm_df['R_Score'] = pd.qcut(rfm_df['recency'].rank(method='dense'), q=4, labels=[1, 2, 3, 4])
    rfm_df['F_Score'] = pd.cut(rfm_df['frequency'], bins=[0, 1, 2, 5, 10], labels=[1, 2, 3, 4], include_lowest=True)
    rfm_df['M_Score'] = pd.qcut(rfm_df['monetary'].rank(method='dense'), q=4, labels=[1, 2, 3, 4])

    rfm_df['RFM_Score'] = rfm_df['R_Score'].astype(str) + rfm_df['F_Score'].astype(str) + rfm_df['M_Score'].astype(str)

    def rfm_segment(score):
        if score in ['444', '434', '344', '443', '433']:
            return 'Best Customers'
        elif score in ['334', '343', '323', '332']:
            return 'Loyal Customers'
        elif score in ['222', '211', '121', '212']:
            return 'Churned Customers'
        else:
            return 'Others'

    rfm_df['Segment'] = rfm_df['RFM_Score'].apply(rfm_segment)
    rfm_df['customer_label'] = rfm_df['customer_id'].str[:5]

    return rfm_df

# Fungsi Load Data
def load_data():
    df = pd.read_csv("dashboard/all_data.csv")
    df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])
    df.sort_values(by="order_purchase_timestamp", inplace=True)
    return df

# Load Data
all_df = load_data()

# Menyiapkan rentang waktu
min_date = all_df["order_purchase_timestamp"].min()
max_date = all_df["order_purchase_timestamp"].max()

# Fungsi untuk menampilkan navigasi sidebar
def sidebar_navigation():
    with st.sidebar:
        st.image("https://github.com/dicodingacademy/assets/raw/main/logo.png")
        start_date, end_date = st.date_input(
            label='Rentang Waktu',
            min_value=min_date,
            max_value=max_date,
            value=[min_date, max_date]
        )
        navigation = st.radio("Pilih Bagian:", [
            "🏠 Beranda", "📈 Tren Pesanan", "🛍️ Kategori Produk", "💳 Metode Pembayaran",
            "⭐ Kepuasan Pelanggan", "📍 Analisis Geografis", "📊 RFM Analysis"
        ])
    return navigation, start_date, end_date

# Ambil nilai yang dikembalikan
selected_section, start_date, end_date = sidebar_navigation()

# Filter data berdasarkan rentang waktu
filtered_df = all_df[(all_df["order_purchase_timestamp"] >= str(start_date)) & 
                     (all_df["order_purchase_timestamp"] <= str(end_date))]

monthly_orders_df = create_monthly_orders_df(filtered_df)
category_sales_df = create_category_sales(filtered_df)
payment_methods_df = create_payment_methods_df(filtered_df)
review_counts_df = create_review_counts_df(filtered_df)
top_cities_df = create_top_cities_df(filtered_df)
top_states_df = create_top_states_df(filtered_df)
rfm_df = create_rfm_df(filtered_df)

def show_order_trends():
    st.header("📈 Tren Jumlah Pesanan Bulanan")
    col1, col2 = st.columns(2)
    with col1:
        min_orders = monthly_orders_df["order_count"].min()
        min_month = monthly_orders_df.loc[monthly_orders_df["order_count"].idxmin(), "order_month"]
        st.metric("Pesanan Terendah", value=min_orders, delta=min_month)

    with col2:
        max_orders = monthly_orders_df["order_count"].max()
        max_month = monthly_orders_df.loc[monthly_orders_df["order_count"].idxmax(), "order_month"]
        st.metric("Pesanan Tertinggi", value=max_orders, delta=max_month)

    # Visualisasi Tren Pesanan Bulanan
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(monthly_orders_df["order_month"], monthly_orders_df["order_count"], marker='o', linestyle='-', linewidth=2, color="#90CAF9")
    ax.set_title("Tren Jumlah Pesanan Bulanan", fontsize=18)
    ax.set_xlabel("Bulan", fontsize=14)
    ax.set_ylabel("Jumlah Pesanan", fontsize=14)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)

    st.pyplot(fig)

def show_product_categories():
    st.header("🛍️ Analisis Kategori Produk")
    # Top categories by sales & revenue
    top_sales_categories = category_sales_df.sort_values('total_units', ascending=False).head(10)
    top_revenue_categories = category_sales_df.sort_values('total_revenue', ascending=False).head(10)

    st.subheader('Kategori Produk Terlaris')

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.barh(top_sales_categories['product_category_name_english'], top_sales_categories['total_units'], color='skyblue')
    ax.set_title('10 Kategori Produk Terlaris', fontsize=12)
    ax.set_xlabel('Jumlah Unit Terjual')
    ax.set_ylabel('Kategori Produk')
    ax.invert_yaxis()
    for bar in bars:
        ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2, f'{int(bar.get_width()):,}', ha='left', va='center', fontsize=9)
    st.pyplot(fig)

    st.subheader('Kategori Produk dengan Pendapatan Tertinggi')

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.barh(top_revenue_categories['product_category_name_english'], top_revenue_categories['total_revenue'], color='lightgreen')
    ax.set_title('10 Kategori Produk dengan Pendapatan Tertinggi', fontsize=12)
    ax.set_xlabel('Total Pendapatan (BRL)')
    ax.set_ylabel('Kategori Produk')
    ax.invert_yaxis()
    for bar in bars:
        ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2, f'R$ {int(bar.get_width()):,}', ha='left', va='center', fontsize=9)
    st.pyplot(fig)

def show_payment_methods():
    st.header("💳 Analisis Metode Pembayaran")
    st.subheader('Frekuensi Penggunaan Metode Pembayaran')

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(payment_methods_df['payment_type'], payment_methods_df['frequency'], color='salmon')

    ax.set_title('Frekuensi Penggunaan Metode Pembayaran', fontsize=16)
    ax.set_xlabel('Metode Pembayaran', fontsize=12)
    ax.set_ylabel('Jumlah Transaksi', fontsize=12)
    ax.set_xticks(range(len(payment_methods_df['payment_type'])))
    ax.set_xticklabels(payment_methods_df['payment_type'], rotation=0)

    # Tambahkan label persentase di atas setiap batang
    for bar, percentage in zip(bars, payment_methods_df['percentage']):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{percentage:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    st.pyplot(fig)

    st.subheader('Analisis Nilai Transaksi per Metode Pembayaran')

    # Hitung nilai transaksi rata-rata
    payment_methods_df['average_value'] = payment_methods_df['total_value'] / payment_methods_df['frequency']

    # Identifikasi metode pembayaran dengan nilai transaksi rata-rata tertinggi
    highest_avg_payment = payment_methods_df.loc[payment_methods_df['average_value'].idxmax()]

    # Tampilkan hasil dalam card-style metric
    st.metric(label="Metode Pembayaran dengan Rata-rata Transaksi Tertinggi",
            value=f"{highest_avg_payment['payment_type']}",
            delta=f"R$ {highest_avg_payment['average_value']:.2f}")

    # Visualisasi nilai transaksi rata-rata per metode pembayaran
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(payment_methods_df['payment_type'], payment_methods_df['average_value'], color='royalblue')

    ax.set_title('Rata-rata Nilai Transaksi per Metode Pembayaran', fontsize=16)
    ax.set_xlabel('Metode Pembayaran', fontsize=12)
    ax.set_ylabel('Rata-rata Nilai Transaksi (BRL)', fontsize=12)
    ax.set_xticks(range(len(payment_methods_df['payment_type'])))
    ax.set_xticklabels(payment_methods_df['payment_type'], rotation=0)

    # Tambahkan label nilai di atas setiap batang
    for bar, avg_value in zip(bars, payment_methods_df['average_value']):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'R$ {avg_value:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    st.pyplot(fig)

def show_customer_reviews():
    st.header("⭐ Analisis Kepuasan Pelanggan")

    # Hitung skor ulasan rata-rata dan distribusi ulasan
    average_score = all_df['review_score'].mean()
    positive_reviews = all_df[all_df['review_score'] >= 4].shape[0]
    neutral_reviews = all_df[all_df['review_score'] == 3].shape[0]
    negative_reviews = all_df[all_df['review_score'] <= 2].shape[0]

    # Menentukan tingkat kepuasan pelanggan
    if average_score >= 4.5:
        satisfaction_level = "Sangat Tinggi 😀"
        color = "green"
    elif average_score >= 4.0:
        satisfaction_level = "Tinggi 😊"
        color = "lightgreen"
    elif average_score >= 3.5:
        satisfaction_level = "Cukup Tinggi 😐"
        color = "gold"
    elif average_score >= 3.0:
        satisfaction_level = "Sedang 😕"
        color = "orange"
    else:
        satisfaction_level = "Rendah 😞"
        color = "red"

    # Tampilkan metrik
    st.metric(label="Skor Ulasan Rata-rata", value=f"{average_score:.2f} / 5")
    st.markdown(f"**Tingkat Kepuasan Pelanggan: <span style='color:{color}; font-size:20px;'>{satisfaction_level}</span>**", unsafe_allow_html=True)

    # Pie Chart Kepuasan Pelanggan
    labels = ['Positif (4-5)', 'Netral (3)', 'Negatif (1-2)']
    sizes = [positive_reviews, neutral_reviews, negative_reviews]
    colors = ['lightgreen', 'gold', 'tomato']

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=140, wedgeprops={'edgecolor': 'black'})
    ax.set_title('Distribusi Sentimen Ulasan Pelanggan', fontsize=14)

    st.pyplot(fig)

def show_geographic_analytics():
    st.header("📍 Analisis Geografis Pembelian")

    top5_cities_percentage = top_cities_df.head(5)['order_count'].sum() / all_df.shape[0] * 100

    # Buat dua kolom untuk visualisasi wilayah
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(7, 6))
        bars = ax.barh(top_cities_df['customer_city'] + ' (' + top_cities_df['customer_state'] + ')', 
                    top_cities_df['order_count'], color='mediumpurple')
        
        ax.set_title('5 Kota dengan Jumlah Pembelian Tertinggi', fontsize=12)
        ax.set_xlabel('Jumlah Pembelian')
        ax.set_ylabel('Kota (Negara Bagian)')
        ax.invert_yaxis()  # Kota teratas di atas

        # Tambahkan label nilai pada setiap bar
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.5, bar.get_y() + bar.get_height()/2, f'{int(width):,}', 
                    ha='left', va='center', fontsize=9)

        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.bar(top_states_df['customer_state'], top_states_df['order_count'], color='teal')
        
        ax.set_title('Distribusi Pembelian per Negara Bagian', fontsize=12)
        ax.set_xlabel('Negara Bagian')
        ax.set_ylabel('Jumlah Pembelian')
        ax.set_xticklabels(top_states_df['customer_state'], rotation=45)

        st.pyplot(fig)

    # Buat tiga kolom untuk ringkasan insight
    colA, colB, colC = st.columns(3)

    with colA:
        st.metric(label="🏆 Kota Teratas", 
                value=f"{top_cities_df.iloc[0]['customer_city']} ({top_cities_df.iloc[0]['customer_state']})", 
                delta=f"{top_cities_df.iloc[0]['order_count']:,} Pembelian")

    with colB:
        st.metric(label="📌 Negara Bagian Teratas", 
                value=f"{top_states_df.iloc[0]['customer_state']}", 
                delta=f"{top_states_df.iloc[0]['order_count']:,} Pembelian")

    with colC:
        st.metric(label="📊 5 Kota Teratas", 
                value=f"{top5_cities_percentage:.1f}%", 
                delta="Dari Total Pembelian")  

def show_rfm_analysis():
    st.header("📊 RFM Analysis - Customer Segmentation")
    # 🎯 **Menampilkan Distribusi Segmen RFM**
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(y=rfm_df['Segment'], order=rfm_df['Segment'].value_counts().index, palette="coolwarm", ax=ax)
    ax.set_title("Distribusi Segmen Pelanggan Berdasarkan RFM", fontsize=14)
    ax.set_xlabel("Jumlah Pelanggan")
    ax.set_ylabel("Segmentasi RFM")
    st.pyplot(fig)

    # 🔹 **Statistik RFM**
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(label="🕒 Recency (Rata-rata)", value=f"{rfm_df['recency'].mean():.1f} Hari")

    with col2:
        st.metric(label="📈 Frequency (Rata-rata)", value=f"{rfm_df['frequency'].mean():.1f} Transaksi")

    with col3:
        st.metric(label="💰 Monetary (Rata-rata)", value=f"R$ {rfm_df['monetary'].mean():,.2f}")

    # 🎯 **Visualisasi Top 5 Pelanggan berdasarkan RFM**
    st.markdown("### 🔥 **Top 5 Best Customers Based on RFM Parameters**")

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(22, 6))

    colors = ["#FF6F61", "#FF6F61", "#FF6F61", "#FF6F61", "#FF6F61"]

    # By Recency
    sns.barplot(y="recency", x="customer_label", 
                data=rfm_df.sort_values(by="recency", ascending=True).head(5), 
                palette=colors, ax=ax[0])
    ax[0].set_title("By Recency (days)", fontsize=16)
    ax[0].set_xlabel("Customer ID")
    ax[0].set_ylabel("Days Since Last Purchase")

    # By Frequency
    sns.barplot(y="frequency", x="customer_label", 
                data=rfm_df.sort_values(by="frequency", ascending=False).head(5), 
                palette=colors, ax=ax[1])
    ax[1].set_title("By Frequency", fontsize=16)
    ax[1].set_xlabel("Customer ID")
    ax[1].set_ylabel("Total Transactions")

    # By Monetary
    sns.barplot(y="monetary", x="customer_label", 
                data=rfm_df.sort_values(by="monetary", ascending=False).head(5), 
                palette=colors, ax=ax[2])
    ax[2].set_title("By Monetary", fontsize=16)
    ax[2].set_xlabel("Customer ID")
    ax[2].set_ylabel("Total Spending (R$)")

    plt.suptitle("Top Customers Based on RFM Parameters", fontsize=20)
    st.pyplot(fig)  

# Menampilkan Header
st.title("E-Commerce Order Dashboard 📊")

if selected_section == "🏠 Beranda":
    show_order_trends()
    show_product_categories()
    show_payment_methods()
    show_customer_reviews()
    show_geographic_analytics()
    show_rfm_analysis()

elif selected_section == "📈 Tren Pesanan":
    show_order_trends()
    
elif selected_section == "🛍️ Kategori Produk":
    show_product_categories()

elif selected_section == "💳 Metode Pembayaran":
    show_payment_methods()
    
elif selected_section == "⭐ Kepuasan Pelanggan":
    show_customer_reviews()

elif selected_section == "📍 Analisis Geografis":
    show_geographic_analytics()
    
elif selected_section == "📊 RFM Analysis":
    show_rfm_analysis()
