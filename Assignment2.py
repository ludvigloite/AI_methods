import numpy as np

def forward(f0,t,O,T,E):

    f = f0

    for k in range(t):
        if E[k] is not None:
            f = np.diag(O[E[k]]) @ T.T @ f
        else:
            f = T.T @ f
        t += 1

    return f


def backward(N,t,O,T,E):

    b = np.ones(T[0].shape)

    for i in range(N-1,t-1,-1):
        #print(i)
        e = E[i]
        O_diag = np.diag(O[e])
        b = T @ O_diag @ b
        #print("b: ", b)
        #print("e: ", e," O: ",O_diag)

    return b


def forward_backward(N,f0,t,O,T,E):

    #fv = []
    #sv = []
    #b = np.ones(t)

    fv = forward(f0,t,O,T,E)
    b = backward(N,t,O,T,E)
    sv = fv * b
    sv = sv / sv.sum()

    return sv




if __name__ == "__main__":
    ##

    runB = False
    runC = False
    runD = True
    runE = False

    #Emission table
    O = np.array([[0.75, 0.2],
                    [0.25, 0.8]])

    # Transition table
    T = np.array([[0.8, 0.3],
                    [0.2, 0.7]]).T

    # Initial estimate
    f0 = np.array([0.5, 0.5])

    # Evidence. Figured i need: Birds = 0, no birds = 1
    E = np.array([0,0,1,0,1,0])

    if runB:

        print("\n--- Task 1b - Filtering ---")

        for t in range(6):
            t += 1

            f = forward(f0,t,O,T,E)
            f = f/f.sum()

            print("\nt = ",t)
            print(f"P(X{t} | e1:{t}) = {f}")
        
    if runC:

        print("\n--- Task 1c - Prediction ---")

        E_pred = [0, 0, 1, 0, 1, 0] + [None]*24

        for t in range(6,30):
            t += 1

            f = forward(f0,t,O,T,E_pred)
            f = f/f.sum()

            print("\nt = ",t)
            print(f"P(X{t} | e1:6) = {f}")

    if runD:

        print("\n--- Task 1d - Smoothing ---")

        N = 6
        #print("O_:", O)
        #print("O_diag:", np.diag(O[0]), np.diag(O[1]))


        for t in range(N):
            sv = forward_backward(N,f0,t,O,T,E)

            print("\nt = ",t)
            print(f"P(X{t} | e1:{N}) = {sv}")




    if runE:

        print("\n--- Task 1e - Most likely sequence ---")


