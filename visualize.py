import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from load_data import load_data

def main():
    columns = '''
                log_days_since_last_purchase
                log_expo_distance_addr
                log_expo_distance_v_sor
                pct_sales_Amt_CC_Payment
                pct_sales_Amt_HPCS_Payment
                sales_amt_3
                visit_9
            '''

    df = load_data('expo_train_data.csv')
    print(df.Mobile1)
    mean = df.mean()
    print(mean[mean > 0.6e9])
    # plt.scatter(list(range(len(mean))), mean)
    # plt.show()



if __name__ == '__main__':
    main()