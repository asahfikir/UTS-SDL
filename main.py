

import marimo

__generated_with = "0.13.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    mo.md(f"""
        # UTS SDL Rijalul Fikri [2024171004]

        Source Code bisa dilihat di https://github.com/asahfikir/UTS-SDL
    """)
    return mo, pd, plt, sns


@app.cell
def _(mo, pd):
    # Load datanya
    df = pd.read_csv('./SewaHunian2019.csv')

    # konversi tipe data agar bisa dianalisa lebih lanjut
    df['price'] = df['price'].astype(float)
    df['last_review'] = pd.to_datetime(df['last_review'])
    df['reviews_per_month'] = pd.to_numeric(df['reviews_per_month'], errors='coerce')

    # Tampilkan dulu informasi sederhana
    mo.md(f"""
    ## Dataset Overview
    - **Jumlah data:** {len(df)}
    - **Kolom:** {', '.join(df.columns)}
    """)

    return (df,)


@app.cell
def _(df):
    df.head()
    return


@app.cell
def _(df, mo):
    # Kelompokkan neighborhood dan hitung metrics nya
    neighborhood_stats = df.groupby('neighbourhood').agg({
        'id': 'count',
        'price': ['mean', 'median'],
        'availability_365': 'mean',
        'number_of_reviews': 'sum'
    }).sort_values(('id', 'count'), ascending=False)

    # Display top 10 neighborhoods
    mo.md(f"""
        ## 10 Lingkungan paling populer untuk disewakan
        Berdasarkan data setelah hasil sortir maka lingkungan paling populer adalah: **Williamsburg, Bedford-Stuyvesant, Harlem, Bushwick, Upper West Side, Hell's Kitchen, East Village, Upper East Side, Crown Heights, dan Midtown**.
        {mo.as_html(mo.ui.table(neighborhood_stats.head(10)))}
    """)

    return


@app.cell
def _(df, mo, plt, sns):
    mo.output.append(mo.md(f"""
        ## Trend Harga berdasarkan Tipe Kamar

        Jika dilihat dari boxplot tipe kamar dapat kita lihat maka tipe `Entire home/apt room_type` memiliki sebaran harga yang jauh lebih tinggi.
    """))

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(
        x='room_type', 
        y='price', 
        data=df, 
        showfliers=False,
        whis=[5, 95],  # Tampilkan 5th-95th percentile, banyak outlier jika tidak di filter
        ax=ax
    )
    ax.set_title('Distribusi Harga (5th-95th Percentile)')

    # Tampilkan
    mo.output.append(mo.mpl.interactive(fig))
    plt.close(fig)
    return


@app.cell
def _(df, mo, pd, plt, sns):
    mo.output.append(
        mo.md(f"""
        ## Trend Harga dari Waktu ke Waktu

        Untuk mengetahui trend harga kita butuh kolom yang berisikan tahun sebagai referensi, salah satu kolom yang bisa digunakan adalah `last_review`. Semua data kita aggregasi berdasarkan kolom ini kemudian kita ambil rata-rata per tahun nya kemudian divisualisasikan.

        Dari hasil visualisasi kita dapat melihat bahwa harga untuk tipe `private room` dan `shared room` cenderung menurun dari tahun ke tahun. Sedangkan tipe `entire home/apt` cenderung lebih stabil kecuali pada tahun 2013 terjadi lonjakan yang sangat signifikan.
        """)
    )
    # Extract year from last_review
    df['year'] = pd.to_datetime(df['last_review']).dt.year

    # Calculate yearly average prices
    yearly_prices = df.groupby(['year', 'room_type'])['price'].mean().reset_index()

    ptfig, ptax = plt.subplots(figsize=(7, 3))
    sns.lineplot(
        x='year',
        y='price',
        hue='room_type',
        data=yearly_prices,
        marker='o',
        ax=ptax
    )
    ptax.set_title('Average Prices by Year and Room Type')
    ptax.set_xlabel('Year')
    ptax.set_ylabel('Average Price ($)')
    ptax.grid(True)
    mo.output.append(mo.mpl.interactive(ptfig))
    plt.close(ptfig)
    return


@app.cell
def _(df, mo):
    # Room type popularity and pricing
    room_type_stats = df.groupby('room_type').agg({
        'id': 'count',
        'price': ['mean', 'median'],
        'number_of_reviews': 'sum',
        'availability_365': 'mean'
    }).sort_values(('id', 'count'), ascending=False)

    mo.output.append(mo.md("""
    ## Popularitas dan Harga berdasarkan Tipe
    """))
    mo.output.append(mo.ui.table(room_type_stats))
    return


@app.cell
def _(df, mo, plt, sns):
    mo.output.append(mo.md(f"""
        ## Matriks Korelasi
        Dari matrik dibawah ini kita dapat melihat bahwa terdapat korelasi negatif lemah untuk jumlah review dan review per bulan dengan harga. Hal ini menandakan bahwa listing dengan harga yang lebih tinggi cenderung untuk meninggalkan jumlah review dan review bulanan yang lebih sedikit.
        Terdapat juga korelasi positif yang cukup lemah antara harga dengan variabel seperti `calculated_host_listings_count`, `availability_365`, dan `minimum_nights`. Dari sini dapat kita simpulkan bahwa harga yang lebih tinggi sedikit berkorelasi dengan host mempunyai lebih banyak listing, availability yang lebih banyak serta minimum stay yang lebih panjang.
    """))

    corr_matrix = df[['price', 'number_of_reviews', 'reviews_per_month', 
                     'calculated_host_listings_count', 'availability_365', 
                     'minimum_nights']].corr()

    cmfig, cmax = plt.subplots(figsize=(8, 5))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    cmax.set_title('Correlation Matrix')
    mo.output.append(mo.mpl.interactive(cmfig))
    return


@app.cell
def _(df, mo):
    # Calculate investment metrics
    investment_metrics = df.groupby('neighbourhood').agg({
        'price': ['mean', 'count'],
        'availability_365': 'mean',
        'number_of_reviews': 'sum'
    }).sort_values(('price', 'mean'), ascending=False)

    # Create composite score (higher price, lower availability, more reviews)
    investment_metrics['score'] = (
        investment_metrics[('price', 'mean')] * 
        investment_metrics[('number_of_reviews', 'sum')] / 
        investment_metrics[('availability_365', 'mean')]
    )

    top_investment = investment_metrics.sort_values('score', ascending=False).head(10)

    mo.output.append(mo.md("""
        ## Top 10 Lingkungan yang berpotensi bagus untuk investasi
    
        Berdasarkan harga, permintaan (diambil dari jumlah review), dan ketersediaan
    """))
    mo.output.append(mo.ui.table(top_investment))
    return


@app.cell
def _(df, mo):
    # Demand analysis (reviews as proxy for demand)
    demand_analysis = df.groupby('neighbourhood').agg({
        'number_of_reviews': 'sum',
        'id': 'count'
    }).sort_values('number_of_reviews', ascending=False)

    demand_analysis['reviews_per_listing'] = demand_analysis['number_of_reviews'] / demand_analysis['id']

    mo.output.append(mo.md("""
    ## Lingkungan berdasar Permintaan (Diukur dari total reviews dan reviews per listing)
    """))
    mo.output.append(mo.ui.table(demand_analysis.head(10)))
    return


@app.cell
def _(df, mo, plt, sns):
    rfig, rax = plt.subplots(figsize=(6, 2))
    sns.scatterplot(x='price', y='reviews_per_month', data=df)
    rax.set_title('Harga vs. Jumlah Review per Bulan')
    plt.xlim(0, 500)  # Focus on majority of prices
    mo.output.append(mo.md(f'''
        ## Ranking
        Karena kita tidak memiliki kolom ranking secara explisit jadi kita menggunakan `reviews_per_month` sebagai basis perhitungan. Jika dilihat dari graph maka dapat kita lihat harga lebih rendah cenderung memiliki tingkat review yang lebih tinggi bisa kita asumsikan lebih populer.
    '''))
    mo.output.append(rfig);
    plt.close(rfig)
    return


@app.cell
def _(df, mo):
    # Reviews berdasarkan lingkungan dan jenis rumah
    review_patterns = df.pivot_table(
        index='neighbourhood',
        columns='room_type',
        values='number_of_reviews',
        aggfunc='sum'
    ).sort_values('Entire home/apt', ascending=False).head(10)

    mo.output.append(mo.md("""
    ## Pola Review berdasarkan Lingkungan dan Tipe (Top 10 lingkungan berdasarkan review)
    """))
    mo.output.append(mo.ui.table(review_patterns))
    return


@app.cell
def _(df, mo):
    # Untuk turis: high reviews, good availability, reasonable price
    # Aggregasi statistiknya
    grouped = df.groupby('neighbourhood').agg(
        total_reviews=('number_of_reviews', 'sum'),
        mean_reviews_per_month=('reviews_per_month', 'mean'),
        mean_price=('price', 'mean'),
        mean_availability=('availability_365', 'mean')
    )

    # Hitung tourist score
    grouped['tourist_score'] = (
        (grouped['total_reviews'] * grouped['mean_reviews_per_month']) /
        (grouped['mean_price'] * (grouped['mean_availability'] + 1))
    )

    # Urutkan tourist score
    tourist_score = grouped['tourist_score'].sort_values(ascending=False)

    # Untuk host: high price, high demand, low availability
    grouped_host = df.groupby('neighbourhood').agg(
        mean_price=('price', 'mean'),
        total_reviews=('number_of_reviews', 'sum'),
        mean_availability=('availability_365', 'mean')
    )

    # Hitung host score
    grouped_host['host_score'] = (
        (grouped_host['mean_price'] * grouped_host['total_reviews']) /
        (grouped_host['mean_availability'] + 1)
    )

    # Urutkan host score
    host_score = grouped_host['host_score'].sort_values(ascending=False)

    mo.md(f"""
    ## Lingkungan Terbaik untuk Turis
    (Top 5 based on reviews, price, and availability)\n
    1. {tourist_score.index[0]}\n
    2. {tourist_score.index[1]}
    3. {tourist_score.index[2]}
    4. {tourist_score.index[3]}
    5. {tourist_score.index[4]}

    ## Lingkungan Terbaik untuk Host
    (Top 5 based on price and demand)\n
    1. {host_score.index[0]}
    2. {host_score.index[1]}
    3. {host_score.index[2]}
    4. {host_score.index[3]}
    5. {host_score.index[4]}
    """)
    return


@app.cell
def _(df, mo):
    # Variasi harga berdasar lingkungan
    price_variance = df.groupby('neighbourhood')['price'].agg(['mean', 'std']).sort_values('std', ascending=False)

    mo.output.append(mo.md("""
    ## Lingkungan dengan Variasi Harga Tertinggi
    (Mengindikasikan opsi rental yang beragam)
    """))
    mo.output.append(mo.ui.table(price_variance.head(10)))
    return


@app.cell
def _(df, mo):
    # Host analysis
    host_analysis = df.groupby('host_id').agg({
        'id': 'count',
        'price': 'mean',
        'number_of_reviews': 'sum'
    }).sort_values('id', ascending=False)

    mo.output.append(mo.md("""
    ## Top Hosts berdasar Jumlah Listings
    """))
    mo.output.append(mo.ui.table(host_analysis.head(10)))
    return


@app.cell
def _(mo):
    mo.md(f"""
    ## Kesimpulan

    1. **Area paling Populer**: **Williamsburg, Bedford-Stuyvesant, Harlem, Bushwick, Upper West Side, Hell's Kitchen, East Village, Upper East Side, Crown Heights, dan Midtown** sering muncul sebagai top lists untuk jumlah listing dan jumlah review.

    2. **Pola Harga**: 
       - homes/apartments memiliki harga tertinggi
       - Private rooms adalah tipe listing yang paling umum
       - Harga menunjukkan naik dan turun namun tidak ada trend yang cukup jelas (terbilang stabil), kecuali di 2013.

    3. **Tipe Rumah**: 
       - Private rooms lebih banyak secara jumlah namun tipe homes menghasilkan revenue lebih banyak
       - Shared rooms adalah yang paling jarang dan umumnya murah

    4. **Korelasi Harga**: 
       - Korelasi negatif lemah untuk jumlah review dan review per bulan dengan harga. Listing dengan harga yang lebih tinggi cenderung untuk meninggalkan jumlah review dan review bulanan yang lebih sedikit.
       - Korelasi positif yang cukup lemah antara harga dengan variabel seperti `calculated_host_listings_count`, `availability_365`, dan `minimum_nights`. Dapat disimpulkan bahwa harga yang lebih tinggi sedikit berkorelasi dengan host mempunyai lebih banyak listing, availability yang lebih banyak serta minimum stay yang lebih panjang.

    5. **Potensi Investasi**: Lingkungan dengan demand tinggi (jumlah review), availability dan harga menunjukkan potensi terbaik investasi.

    6. **Preferensi Turis vs Host**: Turis cenderung memilih lingkungan dengan availability yang baik dan harga bersahabat, sementara host di area dengan banyak permintaan (jumlah review) dan harga tinggi.
    """)
    return


if __name__ == "__main__":
    app.run()
