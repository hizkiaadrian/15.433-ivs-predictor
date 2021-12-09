import numpy as np
import pandas as pd
from matplotlib import cm
import matplotlib.pyplot as plt
from scipy.optimize import minimize, Bounds
from scipy.stats import norm
import statsmodels.api as sm

class IVS:
    def __init__(self, date, df):
        self.date = date
        self.options = df.reset_index(drop=True)
        self.grid = self.get_ivs_grid(df)
        self.surface = self.fit_surface()
        
    def fit_surface(self):
        raise NotImplementedError
        
    def plot(self, grid=True, surface=True):
        if grid:
            assert self.grid is not None
        if surface:
            assert self.surface is not None
            
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        if surface:
            ax.plot_surface(self.surface['x'], self.surface['t'], np.sqrt(self.surface['sigma']), cmap=cm.coolwarm, linewidth=0, alpha=0.5)
        if grid:
            ax.scatter3D(self.grid['x'], self.grid['t'], self.grid['sigma'], s=1)
        ax.set_xlabel('x'); ax.set_ylabel('t'), ax.set_zlabel(r'$\sigma$');
        ax.set_title(f'Plot of IVS on {self.date}')
        ax.text2D(0.95, 0.95, f"MAPE={round(self.mape(), 2)}%", transform=ax.transAxes,
                  bbox=dict(boxstyle="square,pad=0.3",fc="yellow")
                 )
        
    def get_ivs_grid(self, df):
        grid = pd.DataFrame(data = {'x': np.log(df.K / (df.S * np.exp(df.r*df.t)))/np.sqrt(df.t),
                                    't': df.t,
                                    'sigma': df.sigma
                                   })
        
        grid = grid[(grid.x >= -0.2) & (grid.x <= 0.2) & (grid.sigma >= 0.02) & (grid.t > 7/365)]
        
        return grid
    
    def mape(self):
        raise NotImplementedError
        
class SVI(IVS):
    def __init__(self, date, df):
        super().__init__(date, df)
        
        self.grid['vega'] = df.apply(SVI.calc_vega, axis=1)
        self.grid = self.grid.sort_values(['t','x']).reset_index(drop=True)
        
        self.fit_surface()
        
    def fit_surface(self):
        self.sviParams = pd.DataFrame(columns=['a','b','rho','m','sigma'])
        for t in self.grid.t.unique():
            try:
                temp = self.grid[self.grid.t == t]
                params = SVI.quasiExplicitSVIfit(temp.x, temp.sigma ** 2, np.diag(temp.vega))
                self.sviParams.loc[t] = {'a': params[0], 'b': params[1], 'rho': params[2], 'm': params[3], 'sigma': params[4]}
            except:
                continue
                
        x = np.linspace(-0.2, 0.2, 10)
        t = self.sviParams.index.values
        xx, tt = np.meshgrid(x,t)
        zz = np.zeros_like(xx)
        for i in range(xx.shape[0]):
            for j in range(xx.shape[1]):
                zz[i,j] = SVI.rawSVI(xx[i,j], self.sviParams.loc[tt[i,j]])
        self.surface = {'x': xx, 't': tt, 'sigma': zz}
                
    def plot_slices(self):
        assert self.sviParams is not None
        
        rows = int((len(self.sviParams)+len(self.sviParams)%2)/2)
        cols = 2

        fig,ax = plt.subplots(rows,cols,figsize=(12,rows*3+2),
                              constrained_layout=True,gridspec_kw = dict(width_ratios=[5]*cols,
                                  height_ratios=[2]*rows))
        fig.suptitle(f'Options Implied Vol SVI fit ({self.date})',fontsize=24)

        for i, (t, row) in enumerate(self.sviParams.iterrows()):
            x = self.grid[self.grid.t == t].x
            sigma = self.grid[self.grid.t == t].sigma
            row_num = int((i-i%2)/2)
            col_num = int(i%2)
            ax[row_num,col_num].plot(x,np.sqrt(SVI.rawSVI(x, row.values)))
            ax[row_num,col_num].scatter(x, sigma,color='black',s=1)
            ax[row_num,col_num].set_xlabel('Moneyness')
            ax[row_num,col_num].set_ylabel('Implied volatility')
            ax[row_num,col_num].set_title(f'T={round(t,4)}')
        
    def rawSVI(x, sviParams):
        a, b, rho, m, sigma = sviParams
        return a + b * (rho * (x-m) + np.sqrt((x-m)**2 + sigma**2))
    
    def findA_B_rho(m, sigma, x, imp_var, Weight_matrix):
        y = (x - m)/sigma
        z = np.sqrt(y**2 + 1)

        X = np.append(np.append([np.ones(x.shape[0])],[y],axis=0),[z],axis=0).transpose()
        W = Weight_matrix
        Xp = X.transpose()
        beta = np.matmul(np.matmul(np.linalg.inv(np.matmul(Xp,X)),Xp),imp_var)
        beta = np.matmul(np.matmul(np.matmul(np.linalg.inv(np.matmul(np.matmul(Xp,W),X)),Xp),W),imp_var)

        a = beta[0]
        b = beta[2]/sigma
        rho = beta[1]/b/sigma

        return [a,b,rho]
    
    def f_quasi(params, x,imp_var,Weight_matrix):
        m, sigma = params
        a, b, rho = SVI.findA_B_rho(m, sigma, x, imp_var, Weight_matrix)
        return np.linalg.norm(SVI.rawSVI(x,[a, b, rho, m, sigma]) - imp_var)
    
    def quasiExplicitSVIfit(x, imp_var, Weight_matrix):
        m,sigma = minimize(SVI.f_quasi, [0.1,0.1], method='Nelder-Mead', args=(x,imp_var,Weight_matrix),
                           bounds=Bounds([-np.inf,0],[np.inf,np.inf])).x
        a,b,rho = SVI.findA_B_rho(m, sigma, x, imp_var, Weight_matrix)

        return [a,b,rho,m,sigma]
    
    def calc_vega(row):
        d1 = (np.log(row.S / row.K) + (row.r + row.sigma ** 2 / 2) * row.t) / (row.sigma * np.sqrt(row.t))

        return row.S * norm.pdf(d1) * np.sqrt(row.t)
    
    def mape(self):
        actual = self.grid.sigma
        pred = self.grid.apply(lambda row: np.sqrt(SVI.rawSVI(row.x, self.sviParams.loc[row.t])), axis=1)
        
        return np.mean(np.abs((actual - pred) / actual)) * 100
    
class Cubic(IVS):
    def __init__(self, date, df):
        super().__init__(date, df)
        
        self.fit_surface()
        
    def fit_surface(self):
        self.params = pd.DataFrame(columns=['x0','x1','x2','x3'])
        for t in self.grid.t.unique():
            try:
                temp = self.grid[self.grid.t == t]
                params = Cubic.fit_slice(temp.x, temp.sigma)
                self.params.loc[t] = {'x0': params[0], 'x1': params[1], 'x2': params[2], 'x3': params[3]}
            except:
                continue
                
        x = np.linspace(-0.2, 0.2, 10)
        t = self.params.index.values
        xx, tt = np.meshgrid(x,t)
        zz = np.zeros_like(xx)
        for i in range(xx.shape[0]):
            for j in range(xx.shape[1]):
                zz[i,j] = Cubic.get_vol(xx[i,j], self.params.loc[tt[i,j]]) ** 2
        self.surface = {'x': xx, 't': tt, 'sigma': zz}
        
    def plot_slices(self):
        assert self.params is not None
        
        rows = int((len(self.params)+len(self.params)%2)/2)
        cols = 2

        fig,ax = plt.subplots(rows,cols,figsize=(12,rows*3+2),
                              constrained_layout=True,gridspec_kw = dict(width_ratios=[5]*cols,
                                  height_ratios=[2]*rows))
        fig.suptitle(f'Options Implied Vol Cubic fit ({self.date})',fontsize=24)

        for i, (t, row) in enumerate(self.params.iterrows()):
            x = self.grid[self.grid.t == t].x
            sigma = self.grid[self.grid.t == t].sigma
            row_num = int((i-i%2)/2)
            col_num = int(i%2)
            ax[row_num,col_num].plot(x, x.apply(lambda p: Cubic.get_vol(p, self.params.loc[t])))
            ax[row_num,col_num].scatter(x, sigma,color='black',s=1)
            ax[row_num,col_num].set_xlabel('Moneyness')
            ax[row_num,col_num].set_ylabel('Implied volatility')
            ax[row_num,col_num].set_title(f'T={round(t,4)}')
    
    def get_vol(x, params):
        return params @ np.r_[1, x, x**2, x**3]
    
    def fit_slice(x, sigma):
        exog = sm.add_constant(np.c_[x, x**2, x**3])
        return sm.OLS(sigma, exog).fit().params
    
    def mape(self):
        actual = self.grid.sigma
        pred = self.grid.apply(lambda row: Cubic.get_vol(row.x, self.params.loc[row.t]), axis=1)
        return np.mean(np.abs((actual - pred) / actual)) * 100
    
class PolynomialSurface(IVS):
    def __init__(self, date, df):
        super().__init__(date, df)
        
        self.fit_surface()
        
    def fit_surface(self):
        endog = np.log(self.grid.sigma)
        exog = sm.add_constant(np.c_[self.grid.x, self.grid.x ** 2, self.grid.t, self.grid.x * self.grid.t])
        
        res = sm.OLS(endog, exog).fit()
        
        self.res = res
        self.params = res.params
        
        x = np.linspace(-0.2, 0.2, 10)
        t = self.grid.t.unique()
        xx, tt = np.meshgrid(x,t)
        zz = np.zeros_like(xx)
        
        for i in range(xx.shape[0]):
            for j in range(xx.shape[1]):
                zz[i,j] = np.exp(self.params @ np.r_[1, xx[i,j], xx[i,j]**2, tt[i,j], xx[i,j]*tt[i,j]]) ** 2
        self.surface = {'x': xx, 't': tt, 'sigma': zz}
        
    def mape(self):
        actual = self.grid.sigma
        pred = self.grid.apply(lambda row: np.exp(self.params @ np.r_[1, row.x, row.x ** 2, row.t, row.x * row.t]), axis=1)
        return np.mean(np.abs((actual - pred) / actual)) * 100