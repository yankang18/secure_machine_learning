import socketserver

from legacy_code.common_data import *


class ThreadedTCPRequestHandler(socketserver.BaseRequestHandler):

    def __recv__(self):
        data = b''
        while True:
            # print ('Receiving...')
            recv_data = self.request.recv(1024 * 20)
            if len(recv_data) != 0:
                data = data + recv_data
            else:
                return data

    def handle(self):
        self.data = self.__recv__()
        q.put(self.data)
        # cur_thread = threading.current_thread()
        # cur_str = "{}: {}".format(cur_thread.name, data)
        s.release()


class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    pass


def p1_local_generate(dimension_para):
    global p1_beaver_data
    '''
    Gen local Beaver triples
    '''
    print('StartToGenBeaverTriples')
    n = dimension_para.n
    d = dimension_para.d
    t = dimension_para.t3
    B = dimension_para.B

    M1 = generate_random_matrix(n, d)
    M11 = generate_random_matrix(n, d)
    M10 = M1 - M11

    N1 = generate_random_matrix(d, t)
    N11 = generate_random_matrix(d, t)
    N10 = N1 - N11

    P1 = generate_random_matrix(B, t)
    P11 = generate_random_matrix(B, t)
    P10 = P1 - P11

    # Keep M1, M10, M11; N1, N10, N11; P1, P10, P11
    p1_beaver_data = P1BeaverData(M1, M10, M11, N1, N10, N11, P1, P10, P11)
    send_to_P0(P1ToP0(M10, N10, P10))
    print('P1SentToP0[OK]')

    send_to_Carlo(P1ToCarlo(M11, N11, P11, dimension_para))
    print('P1SentToCarlo[OK]')
    print('N1.dim=%s' % (str(p1_beaver_data.N1.shape)))


def train():
    global dimension_para
    global p1_beaver_data
    global E
    global y1
    global w1
    global X1
    global E1
    global w1
    global observe_lst
    s_train_loop.acquire()
    n = dimension_para.n
    d = dimension_para.d
    t = dimension_para.t
    B = dimension_para.B
    epoches = n // B
    w1 = init_w(d)
    logging.info('BeforeTrain')
    logging.debug('Init w1={}'.format(w1))
    for j in range(t):
        k = j % epoches
        f1 = w1 - p1_beaver_data.N1[:, j:j + 1]

        # reconstruct fj
        # send message to reconstruct fj
        send_to_P0(P1ReconstructFj(j, f1))
        logging.info('SendToP0ReconstructFj')
        s_fj_train.acquire()
        p0_reconstruct_fj = q_fj_train.get(block=True)
        if p0_reconstruct_fj.j != j:
            print('WARN****WithDifferentj*****')
            print('Remote j={}, cur j={}'.format(p0_reconstruct_fj.j, j))
        f = f1 + p0_reconstruct_fj.f0
        y_star1 = np.array(E[k * B:k * B + B]).dot(f) + \
                  np.array(p1_beaver_data.M1[k * B:k * B + B]).dot(f) + \
                  np.array(E[k * B:k * B + B]).dot(p1_beaver_data.N1[:, j:j + 1]) + \
                  np.array(p1_beaver_data.O1[:, j:j + 1])

        # y_star1 = np.array(p1_beaver_data.M1[k*B:k*B+B]).dot(f)+\
        # np.array(E[k*B:k*B+B]).dot(np.array(p1_beaver_data.N1[:,j:j+1]))+\
        # np.array(p1_beaver_data.O1[:,j:j+1])

        observe_lst.append(y_star1)
        y_star1 = 0.5 + 0.5 * approx_rate * y_star1
        s1 = y_star1 - np.array(y1[k * B:k * B + B])
        f_star1 = s1 - p1_beaver_data.P1[:, j:j + 1]
        send_to_P0(P1ReconstructFStarj(j, f_star1))
        logging.info('SendToP0ReconstructFStarj')
        s_fstarj_train.acquire()
        p0_reconstruct_fstarj = q_fstarj_train.get(block=True)
        if p0_reconstruct_fstarj.j != j:
            print('WARN****WithDifferentj 2*****')
            print('Remote j={}, cur j={}'.format(p0_reconstruct_fstarj.j, j))
        f_star = f_star1 + p0_reconstruct_fstarj.fstar0

        g1 = np.array(E[k * B:k * B + B]).transpose().dot(f_star) + \
             np.array(p1_beaver_data.M1[k * B:k * B + B]).transpose().dot(f_star) + \
             np.array(E[k * B:k * B + B]).transpose().dot(p1_beaver_data.P1[:, j:j + 1]) + \
             np.array(p1_beaver_data.Q1[:, j:j + 1])

        # g1 = np.array(p1_beaver_data.M0[k*B:k*B+B]).transpose().dot(f_star)+\
        # np.array(E[k*B:k*B+B]).transpose().dot(p0_beaver_data.P0[:,j:j+1])+\
        # np.array(p0_beaver_data.Q0[:,j:j+1])

        w1 = w1 - learning_rate / B * g1
        logging.info('FinishIteration{}'.format(j))
    # Reconstruct w
    send_to_P0(P1ReconstructW(w1))
    logging.info('FinishTraining')
    logging.debug('Final w1={}'.format(w1))
    print('SendRecontructWToP0')
    # write_to_file('P1.csv', np.concatenate(observe_lst,axis=1))


def p1_deal_data(data):
    # TODO write more comments for this part.
    global dimension_para
    global X1
    global E1
    global E
    global y1
    global w1
    global Xtest
    global Ytest
    if data.data_type == 'DistributeData':
        # Received data share from the data source.
        dimension_para = data.dim_para
        X1 = data.X
        y1 = data.y
        Xtest = data.Xtest
        Ytest = data.Ytest
        print('ReceivedDataFromDataSource n=%d,d=%d,t=%d, B=%d' \
              % (dimension_para.n, dimension_para.d, dimension_para.t, dimension_para.B))
        # Send data to P0 and Carlo to generate the Beaver triples.
        p1_local_generate(dimension_para)
        return True
    elif data.data_type == 'P0ToP1':
        print('ReceiveDataFromP0')
        q_data.put(data)
        s_data.release()
        return True
    elif data.data_type == 'CarloToP1':
        print('ReceiveDataFromCarlo')
        q_data.put(data)
        s_data.release()
        return True
    elif data.data_type == 'P0StartTraining':
        # Received starting training signal from P0
        s_start_train.acquire()
        print('ReceiveStartTrainingFromP0')
        # start training
        # reconstruct matrix E
        E1 = X1 - p1_beaver_data.M1
        send_to_P0(P1ReconstructE(E1))
        s_E1.release()
        logging.info('E1IsAvailable')
        return True
    elif data.data_type == 'P0ReconstructE':
        print('ReceiveRecontructEFromP0')
        s_E1.acquire()
        logging.debug('E1.shape={}'.format(E1.shape))
        logging.debug('data.E0.shape={}'.format(data.E0.shape))
        E = E1 + data.E0
        # start training now
        s_train_loop.release()
        return True
    elif data.data_type == 'P0ReconstructFj':
        logging.info('ReceiveP0ReconstructFj')
        q_fj_train.put(data)
        s_fj_train.release()
        logging.debug('AllowToUseTheFjData')
        return True
    elif data.data_type == 'P0ReconstructFStarj':
        logging.info('ReceiveP0ReconstructFStarj')
        q_fstarj_train.put(data)
        s_fstarj_train.release()
        return True
    elif data.data_type == 'P0ReconstructW':
        print('ReceiveP0ReconstructW')
        w = data.w0 + w1
        print('RecoverWNow')
        server.shutdown()
        accuracy = compute_accuracy(Xtest, Ytest, w)
        print('accuracy=%.5f' % (accuracy))
        logging.info('accuracy=%.5f' % (accuracy))
        print(cur_time_str())
        logging.info(cur_time_str())
        return False
    else:
        print('NoSuchDataType')
        print(data.data_type)
        return False


class P1DealThread(threading.Thread):
    '''
    Deal received data thread
    '''

    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        while True:
            s.acquire()
            data = load_obj(q.get(block=True))
            logging.info('ReceivedDataType={}'.format(data.data_type))
            if not p1_deal_data(data):
                break
        print('FinishP1DealThread')


def P1_deal_partial_beaver_data(data1, data2):
    # TODO consider this part again.
    global p1_beaver_data
    if data1.data_type == 'P0ToP1' and data2.data_type == 'CarloToP1':
        p0_to_p1 = data1
        c_to_p1 = data2
    elif data1.data_type == 'CarloToP1' and data2.data_type == 'P0ToP1':
        c_to_p1 = data1
        p0_to_p1 = data2
    else:
        print('IncorrectDataType')
        exit(-1)
    # do the multiplication now then send to P0 start signal
    n = dimension_para.n
    d = dimension_para.d
    t = dimension_para.t
    B = dimension_para.B
    O1 = np.zeros((B, t))
    Q1 = np.zeros((d, t))
    epoches = n // B
    M1, M10, M11 = p1_beaver_data.M1, p1_beaver_data.M10, p1_beaver_data.M11
    N1, N10, N11 = p1_beaver_data.N1, p1_beaver_data.N10, p1_beaver_data.N11
    P1, P10, P11 = p1_beaver_data.P1, p1_beaver_data.P10, p1_beaver_data.P11
    for j in range(t):
        k = j % epoches
        O1[:, j:j + 1] = np.array(M1[k * B:k * B + B]).dot(np.array(N1[:, j:j + 1])) + \
                         np.array(p0_to_p1.M01[k * B:k * B + B]).dot(np.array(N11[:, j:j + 1])) + \
                         np.array(c_to_p1.O1[:, j:j + 1]) + \
                         np.array(M11[k * B: k * B + B]).dot(np.array(p0_to_p1.N01[:, j:j + 1]))

        Q1[:, j:j + 1] = np.array(M1[k * B:k * B + B]).transpose().dot(np.array(P1[:, j:j + 1])) + \
                         np.array(p0_to_p1.M01[k * B:k * B + B]).transpose().dot(np.array(P11[:, j:j + 1])) + \
                         np.array(c_to_p1.Q1[:, j:j + 1]) + \
                         np.array(M11[k * B:k * B + B]).transpose().dot(np.array(p0_to_p1.P01[:, j:j + 1]))

    p1_beaver_data.O1 = O1
    p1_beaver_data.Q1 = Q1
    send_to_P0(P1StartTraining())
    print('FinishGenBeaverTriples&SendStartTrainingSignalToP1')
    s_start_train.release()
    print('M1.shape=%s,N1.shape=%s,O1.shape=%s,P1.shape=%s,Q1.shape=%s' \
          % (p1_beaver_data.M1.shape, p1_beaver_data.N1.shape, \
             p1_beaver_data.O1.shape, p1_beaver_data.P1.shape, p1_beaver_data.Q1.shape))

    logging.info('P1M1={}'.format(p1_beaver_data.M1))
    logging.info('P1N1={}'.format(p1_beaver_data.N1))
    logging.info('P1O1={}'.format(p1_beaver_data.O1))
    logging.info('P1P1={}'.format(p1_beaver_data.P1))
    logging.info('P1Q1={}'.format(p1_beaver_data.Q1))


class WaitBeaverDataThread(threading.Thread):
    # wait for the Beaver triple data thread
    def run(self):
        s_data.acquire()
        s_data.acquire()
        data1 = q_data.get(block=True)
        data2 = q_data.get(block=True)
        # Now deal with both data
        P1_deal_partial_beaver_data(data1, data2)


q = Queue()
s = threading.Semaphore(value=0)
# s_data and q_data are used to wait for the data from Carlo and P0.
# WaitBeaverDataThread waits for s_data and does the remaining 
# computation to generate the Beaver triples.
s_data = threading.Semaphore(value=0)
q_data = Queue()
# s_start_train is used to wait for the completion of generation ofBeaver triples.
s_start_train = threading.Semaphore(value=0)

s_train = threading.Semaphore(value=0)
q_train = Queue()
# s_E1 is used to wait for the starting the reconstructE operation.
s_E1 = threading.Semaphore(value=0)
# s_train_loop is used to wait for the completion of the reconstructE operation.
s_train_loop = threading.Semaphore(value=0)
# s_fj_train is used to wait for the reconstructFj operation.
s_fj_train = threading.Semaphore(value=0)
q_fj_train = Queue()
s_fstarj_train = threading.Semaphore(value=0)
q_fstarj_train = Queue()
dimension_para = None
X1 = None
y1 = None
E1 = None
E = None
w1 = None
# p0_data = None
Carlo_data = None
server = None
Xtest = None
Ytest = None
p1_beaver_data = None
observe_lst = []
if __name__ == '__main__':
    init_logging('logger_P1.log')
    print(cur_time_str())
    logging.info(cur_time_str())
    HOST, PORT = read_config().P1
    print('P1ServerInfo {}:{}'.format(HOST, PORT))
    server = ThreadedTCPServer((HOST, PORT), ThreadedTCPRequestHandler)
    # receive data thread
    server_thread = threading.Thread(target=server.serve_forever)
    # deal received data thread
    p1_deal_thread = P1DealThread()
    # training thread
    train_thread = threading.Thread(target=train)
    # wait for the initial data thread
    wait_beaver_data_thread = WaitBeaverDataThread()
    thread_lst = (server_thread, p1_deal_thread, train_thread, wait_beaver_data_thread)
    start_all_threads(thread_lst)
