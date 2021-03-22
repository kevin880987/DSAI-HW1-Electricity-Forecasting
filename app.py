
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import inspect
import pickle

from datetime import datetime, timedelta
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import StandardScaler

from keras.models import Sequential
from keras.layers import Dense, Dropout, GRU
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

# self=Preprocessor()
class Preprocessor():
    def __init__(self):
        pass
 
    # df=training_df
    def amortize(self, df):
        """
        Input a day-wise sparse dataframe.
        Return an amortized dataframe.
    
        Parameters
        ----------
        df : dataframe
            A sparse dataframe with date as its index.
            
            e.g.
              DATE  Brent Oil Futures Historical Data - Price
        2010-01-01                                        NaN
        2010-01-02                                        NaN
        2010-01-03                                        NaN
        2010-01-04                                      80.12
        2010-01-05                                      80.59
    
        Par : dictionary
            Costomized parameters imported from 'parameters.py'.
    
        Raises
        ------
        ValueError
            Raised when the amortization contains NaN.
    
        Returns
        -------
        df : dataframe
            A dataframe with no NaN and date as its index.
            
            e.g.
              DATE  Brent Oil Futures Historical Data - Price
        2010-01-01                                      80.12
        2010-01-02                                      80.12
        2010-01-03                                      80.12
        2010-01-04                                      80.12
        2010-01-05                                      80.59
    
        """
        
        display, verbose = True, True
        if display:
            feature_ctr, unab_amort_list = 0, []
    
        df = df.copy()
        
        for col in df.columns:
            # if verbose:
            #     print(col)
    
            index = np.where(df[col].notnull())[0]
            if index.size >= 2:
                amortization = [df[col].iloc[index[0]]] * (index[0] - 0)
                for i in range(len(index)-1):
                    amortization.extend(
                        np.linspace(float(df[col].iloc[index[i]]), 
                                    float(df[col].iloc[index[i+1]]), 
                                    index[i+1]-index[i], endpoint=False)
                        )    
    
                    if np.any(pd.isnull(amortization)):
                        print(i)
                        raise ValueError(f'{col} contains NaN')
    
                amortization.extend(
                    [df[col].iloc[index[i+1]]] * (len(df[col]) - 1 - index[i+1] + 1)
                    )
                        
                df[col] = amortization
                
                # Make sure all values are converted into number
                df[col] = df[col].astype(float)
                
                if np.any(pd.isnull(df[col])):
                    print('null', col)
                    raise ValueError
                
                if display:
                    feature_ctr += 1
                
            elif index.size < 2:
                if display:
                    unab_amort_list.append(col)
                if verbose:
                    print(f'Unable to amortize {col}')
                    
                df.drop(columns=col, inplace=True)
                
        return df

# X, Y=X_train, Y_train
    def build_train(self, X=pd.DataFrame(), Y=pd.DataFrame()):
        train_steps = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        predict_steps = np.array([1, 2, 3, 4, 5, 6, 7, 8]) # must start with 1
        self.predict_steps = predict_steps
        
        # if np.any([x!=y for x, y in zip(X.index, Y.index)]):
        #     raise ValueError('Index Not Compatible')
            
        # Amortize
        X = self.amortize(X)
        Y = self.amortize(Y)

        X_train = pd.DataFrame(index=X.index)
        for step in train_steps:
            X_train[[f'{c} - (-{step})' for c in X.columns]] = X.shift(step)
        # X_train.dropna(axis=0, inplace=True)
        
        Y_train = pd.DataFrame(index=Y.index)
        for step in predict_steps:
            Y_train[[f'{t} - (+{step})' for t in Y.columns]] = Y.shift(-step)
        # Y_train.dropna(axis=0, inplace=True)
        
        # Get index intersection
        index = sorted(list(set(X_train.index) & set(Y_train.index)))
        X_train = X_train.loc[index]
        Y_train = Y_train.loc[index]
        
        return X_train, Y_train

    def shuffle(self, X, Y):     
        # Shuffle index
        index = list(X.index)
        np.random.seed(10)
        np.random.shuffle(index)
        X = X.loc[index]
        Y = Y.loc[index]
        
        return X, Y

# Y=training_df[[target]]
    def fit_transform(self, Y=pd.DataFrame()):
        # For dataframe with unstretched columns and datetime index
        # Y must be a dataframe with one column

        # Decomposing config
        period = 7
        decompose_model = 'additive'

        target = Y.columns[0]
        
        # # Differenciate
        # self.Y_observed = Y.copy() # for inversing
        # Y = Y.diff(periods=1).dropna(axis=0)
        
        # Scale
        self.Y_scaler = StandardScaler().fit(Y)
        Y.loc[:, :] = self.Y_scaler.transform(Y)        

        # Decompose the target
        self.Y_decomposer = seasonal_decompose(Y[target], model=decompose_model, period=period)
        Y[target] = self.Y_decomposer.trend

        self.period = period
        self.target = target
        self.decompose_model = decompose_model
        return Y

# a=self
# self=a.preprocessor
# Y=Y_pred.copy()
# Y=Y_train.copy()
    def inverse(self, Y=pd.DataFrame()):
        # For dataframe with multiple stretched columns and datetime index
        
        # Inverse decomposition
        seasonal = self.Y_decomposer.seasonal
        freq = pd.to_datetime(seasonal.index).inferred_freq
        delta = pd.Timedelta(1, freq)
        seasonal_period = pd.Timedelta(self.period, freq)
        
        for yi in pd.to_datetime(Y.index):
            for ti in pd.to_datetime(seasonal.index):
                if (yi-ti)%seasonal_period<pd.Timedelta(1, freq):
                    break

            index = pd.to_datetime(ti + delta*self.predict_steps).strftime('%Y-%m-%d')

            if self.decompose_model=='additive':
                Y.loc[yi.strftime('%Y-%m-%d'), :] += seasonal.loc[index].values
            elif self.decompose_model=='multiplicative':
                Y.loc[yi.strftime('%Y-%m-%d'), :] *= seasonal.loc[index].values
        
        # Inverse scaling
        Y.loc[:, :] = self.Y_scaler.inverse_transform(Y)

        # # Inverse difference
        # Y.iloc[0, 0] += self.Y_observed.loc[Y.index[0]].values[0]
        # Y.iloc[0, :] = Y.iloc[0, :].cumsum(axis=0)

        # Reshape
        Y.columns = pd.to_datetime(Y.index[0]) + delta*self.predict_steps
        Y.index = [self.target]
        Y = Y.transpose()
        Y.index.name = seasonal.index.name

        return Y

# self=Model()
class Model():
    def __init__(self):
        pass
                    
    def build_model(self, X_train, Y_train): # , X_val, Y_val
        layers = 8 # 4 # 
        units = 16 # 4 # 
        dropout = 0.05
        
        loss = 'mse'
        optimizer = 'adam'
        
        epochs = 20000 # 2 # 
        batch_size = 128
        patience = 100
        
        starttime = datetime.now()
        print()
        print()
        print(inspect.currentframe().f_code.co_name)
        print('\tstart time:', starttime)
        print()

        X_train, Y_train = X_train.dropna(axis=0).copy(), Y_train.dropna(axis=0).copy()
        shape = X_train.shape
        nn_model = Sequential()
        for i, u in zip(range(layers), np.linspace(units, 2, layers)):
            if i<layers-1:
                return_sequences = True
            else:
                return_sequences = False
                
            nn_model.add(GRU(units=int(np.round(max(1, u))), input_shape=(shape[1], 1), 
                          return_sequences=return_sequences))
            nn_model.add(Dropout(dropout))
        
        nn_model.add(Dense(Y_train.shape[1])) # nn_model.add(TimeDistributed(Dense(1)))
        nn_model.compile(loss=loss, optimizer=optimizer)
        nn_model.summary()
        
        X_train_re = np.reshape(X_train.values, (X_train.shape[0], X_train.shape[1], 1))
        Y_train_re = Y_train.values[:, :, np.newaxis]
        # es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=patience) # 
        # mc = ModelCheckpoint('best_model', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
        history = nn_model.fit(X_train_re, Y_train_re, 
                            validation_split=0.2, 
                            epochs=epochs, batch_size=batch_size, 
                            callbacks=[]) # , validation_data=(X_val, Y_val) # es, mc
        
        self.nn_model = nn_model
        
        # # Save model
        # with open(f'nn_model.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
        #     pickle.dump(nn_model, f)

        # Visualize
        plt.plot(history.history['loss'])
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss Function Value')
        plt.legend(['Loss'])
        plt.show()

        endtime = datetime.now()
        print()
        print(inspect.currentframe().f_code.co_name)
        print('\tend time:', endtime)
        print('\ttime consumption:', endtime-starttime)
        print()
        print()
        
    def predict(self, i=None, display=False):
        if i is None:
            display = True
            i = pd.to_datetime(self.X_train.index).argmax()
            
        # # Loade model
        # with open(f'nn_model.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
        #     nn_model = pickle.load(f)
            
        nn_model = self.nn_model
        testing = self.X_train.iloc[[i]].values
        
        # Predict
        # Y_pred = nn_model.predict(self.X_train_re[[i], :, :])
        Y_pred = nn_model.predict(np.reshape(testing, (testing.shape[0], testing.shape[1], 1)))
        Y_pred = pd.DataFrame(Y_pred, index=self.Y_train.index[[i]], 
                              columns=self.Y_train.columns)
        
        # Inverse
        Y_pred = self.preprocessor.inverse(Y=Y_pred)

        if display:
            # Visualize
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
            self.training_df[self.target].plot(ax=ax, legend=True)
            ax.plot(Y_pred, label=True)
            plt.title('Prediction')
            plt.xlabel('Time')
            plt.ylabel('Target Value')
            plt.legend(['Real Y', 'Predicted Y'])
            plt.show()
        
        return Y_pred

# a=self
# self=a
# training_df=df_training.copy()
    def train(self, training_df):
        target = '備轉容量(MW) - MODIFIED.csv'
        freq = pd.to_datetime(training_df.index).inferred_freq
        delta = pd.Timedelta(1, freq)

        # root_fp = os.getcwd() + os.sep
        # training_df = pd.read_csv(root_fp+'training_data.csv', encoding='big5', index_col=0)
        self.training_df = training_df
        self.target = target
        
        # Preprocess data
        X_train = training_df
        self.preprocessor = Preprocessor()
        Y_train = self.preprocessor.fit_transform(training_df[[target]])
        X_train, Y_train = self.preprocessor.build_train(X_train, Y_train) # Build training x y
        X_train, Y_train = self.preprocessor.shuffle(X_train, Y_train)
        self.X_train, self.Y_train = X_train, Y_train 
        
        # Build and fit nn_model
        self.build_model(X_train, Y_train)
        
        # Predict train and visualize
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
        training_df.index = pd.to_datetime(training_df.index)
        training_df[target].plot(ax=ax, legend=True)
        for i in range(X_train.shape[0]):
            # Predict
            Y_pred = self.predict(i)
            ax.plot(Y_pred, label=False)

        plt.title('Training')
        plt.xlabel('Time')
        plt.ylabel('Target Value')
        plt.legend(['Real Y'])
        plt.show()


if __name__ == '__main__':
    # You should not modify this part, but additional argument
    # s are allowed.
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--training', 
                        default='training_data.csv', 
                        help='input training data file name')
    parser.add_argument('--output', 
                        default='submission.csv', 
                        help='output file name')
    args = parser.parse_args()
    
    # The following part is an example.
    # You can modify it at will.
    root_fp = os.getcwd() + os.sep
    df_training = pd.read_csv(root_fp+args.training, encoding='big5', index_col=0)
    model = Model()
    model.train(df_training)

    df_result = model.predict()
    df_result.to_csv(args.output)
