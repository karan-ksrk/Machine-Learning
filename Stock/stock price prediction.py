import csv
from tkinter import *
from tkinter import ttk
import numpy as np
from sklearn.svm import SVR
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure


dates = []
prices = []

####### get csv file form directory #######
def get_data(filename):
    with open(filename, "r") as csvfile:
        csvFileReader = csv.reader(csvfile)
        next(csvFileReader)
        for row in csvFileReader:
            dates.append(int(row[0].split("-")[-1]))
            prices.append(float(row[1]))


####### setting up tkinter window #######
root = Tk()
root.geometry("700x700")
root.resizable(0, 0)
root.configure(background="grey")
root.title("Stock Price Predictor")


##################################################### Menubar ##########################################################################
menubar = Menu(root)
root.config(menu=menubar)
file_ = Menu(menubar)
edit = Menu(menubar)
help_ = Menu(menubar)
menubar.add_cascade(menu=file_, label="File")
menubar.add_cascade(menu=edit, label="Edit")
menubar.add_cascade(menu=help_, label="Help")

####### Setting up Frames #######
frame1 = ttk.Frame(root)
frame2 = ttk.Frame(root, width=300, height=200)
frame3 = ttk.Frame(root)

frame1.grid(row=0, pady=10)
frame2.grid(row=1, padx=100, pady=10)
frame3.grid(row=2, padx=50, pady=10)

##################################################### Frame1 ##########################################################################
companies = ["Apple", "Microsoft", "Facebook", "Google", "Accenture", "Intel"]
company_name = StringVar(frame1)
company_name.set(companies[0])
ttk.Label(frame1, text="Company:", font="Helvetica 10 bold").grid(
    row=0, column=0, pady=10
)
w = OptionMenu(*(frame1, company_name) + tuple(companies))
w.grid(row=0, column=1)
ttk.Button(frame1, text="Draw Graph", command=lambda: predict_graph()).grid(
    row=1, columnspan=2, pady=2
)

##################################################### Frame2 ##########################################################################
fig = Figure(figsize=(5, 4), dpi=100)
s = fig.add_subplot(111)
canvas = FigureCanvasTkAgg(fig, master=frame2)

# to draw graph of stock from csv file by calling get_data function
def predict_graph():
    global svr_lin, svr_rbf, s, fig, canvas, dates, prices
    get_data(company_name.get() + ".csv")
    dates = np.reshape(dates, (len(dates), 1))
    svr_lin = SVR(kernel="linear", C=1e3)
    svr_rbf = SVR(kernel="rbf", C=1e3)
    svr_lin.fit(dates, prices)
    svr_rbf.fit(dates, prices)
    s.clear()
    s.scatter(dates, prices, color="black", label="Data")
    s.plot(dates, svr_lin.predict(dates), color="green", label="Linear model")
    s.plot(dates, svr_rbf.predict(dates), color="red", label="RBF model")
    dates = []
    prices = []
    canvas.draw()
    s.plot(dates, prices)
    canvas.get_tk_widget().grid(row=0)


##################################################### Frame3 ##########################################################################
ttk.Label(frame3, text="Days:", font="Helvetica 10 bold").grid(row=0, column=0, padx=10)
days = ttk.Entry(frame3, width=6, font=("Arial", 10))
days.grid(row=0, column=1)

ttk.Label(frame3, text="kernel:", font="Helvetica 10 bold").grid(
    row=0, column=2, padx=10
)
kernels = ["Linear", "rbf"]
kernel = StringVar(frame3)
kernel.set(kernels[0])
X = OptionMenu(*(frame3, kernel) + tuple(kernels))
X.grid(row=0, column=3)


answer = ttk.Entry(frame3, width=10, font=("Arial", 10))
answer.grid(row=2, columnspan=4, pady=10)
ttk.Button(frame3, text="Predict Price", command=lambda: showanswer()).grid(
    row=1, columnspan=4, pady=6
)

# Show predicted answer on answer widget
def showanswer():
    global answer
    answer.delete(0, END)
    a = predict_price(days.get(), kernel.get())
    answer.insert(0, str(a))
    answer.update()


# predict price
def predict_price(x, kernel):
    x = int(x)
    if kernel == "Linear":
        return svr_lin.predict([[x]])[0]
    else:
        return svr_rbf.predict([[x]])[0]


# first time show a default graph
predict_graph()


# applying some style
style = ttk.Style()
style.theme_use("classic")
# style.theme_use("aqua")
style.configure("TButton", foreground="red", font=("Arial", 16, "bold"))

root.mainloop()
