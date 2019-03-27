import socketserver

from common_data import *


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


def p0_local_generate(dimension_para):
    global p0_beaver_data
    '''
    Gen local Beaver triples
    '''
    print('StartToGenBeaverTriples')
    n = dimension_para.n
    d = dimension_para.d
    t = dimension_para.t
    B = dimension_para.B

    M0 = generate_random_matrix(n, d)
    M00 = generate_random_matrix(n, d)
    M01 = M0 - M00

    N0 = generate_random_matrix(d, t)
    N00 = generate_random_matrix(d, t)
    N01 = N0 - N00

    P0 = generate_random_matrix(B, t)
    P00 = generate_random_matrix(B, t)
    P01 = P0 - P00

    # Keep M0, M00, M01; N0, N00, N01, P0, P00, P01
    p0_beaver_data = P0BeaverData(M0, M00, M01, N0, N00, N01, P0, P00, P01)

    send_to_P1(P0ToP1(M01, N01, P01))
    print('P0SentToP1[OK]')
    send_to_Carlo(P0ToCarlo(M00, N00, P00, dimension_para))
    print('P0SentToCarlo[OK]')
    print('N0.dim=%s' % (str(p0_beaver_data.N0.shape)))


def train():
    global dimension_para
    global p0_beaver_data
    global E
    global y0
    global w0
    global X0
    global E0
    global w0
    global observe_lst
    s_train_loop.acquire()
    n = dimension_para.n
    d = dimension_para.d
    t = dimension_para.t
    B = dimension_para.B
    epoches = n // B
    w0 = init_w(d)
    logging.info('BeforeTrain')
    logging.debug('Init w0={}'.format(w0))
    for j in range(t):
        k = j % epoches

        # TODO: compute logits
        f0 = w0 - p0_beaver_data.N0[:, j:j + 1]
        # reconstruct fj
        # send message to reconstruct fj
        send_to_P1(P0ReconstructFj(j, f0))
        logging.info('SendToP1ReconstructFj')
        s_fj_train.acquire()

        p1_reconstruct_fj = q_fj_train.get(block=True)
        if p1_reconstruct_fj.j != j:
            print('WARN****WithDifferentj*****')
            print('Remote j={}, cur j={}'.format(p1_reconstruct_fj.j, j))
        f = f0 + p1_reconstruct_fj.f1

        y_star0 = np.array(p0_beaver_data.M0[k * B:k * B + B]).dot(f) + \
                  np.array(E[k * B:k * B + B]).dot(p0_beaver_data.N0[:, j:j + 1]) + \
                  np.array(p0_beaver_data.O0[:, j:j + 1])

        # y_star0 = np.array(p0_beaver_data.M0[k*B:k*B+B]).dot(f)+\
        # np.array(E[k*B:k*B+B]).dot(np.array(p0_beaver_data.N0[:,j:j+1]))+\
        # np.array(p0_beaver_data.O0[:,j:j+1])

        observe_lst.append(y_star0)

        # TODO: compute prediction
        y_star0 = 0.5 * approx_rate * y_star0

        # TODO: compute gradients
        s0 = y_star0 - np.array(y0[k * B:k * B + B])
        f_star0 = s0 - p0_beaver_data.P0[:, j:j + 1]
        send_to_P1(P0ReconstructFStarj(j, f_star0))
        logging.info('SendToP1ReconstructFStarj')
        s_fstarj_train.acquire()
        p1_reconstruct_fstarj = q_fstarj_train.get(block=True)
        if p1_reconstruct_fstarj.j != j:
            print('WARN****WithDifferentj 2*****')
            print('Remote j={}, cur j={}'.format(p1_reconstruct_fstarj.j, j))
        f_star = f_star0 + p1_reconstruct_fstarj.fstar1

        g0 = np.array(p0_beaver_data.M0[k * B:k * B + B]).transpose().dot(f_star) + \
             np.array(E[k * B:k * B + B]).transpose().dot(p0_beaver_data.P0[:, j:j + 1]) + \
             np.array(p0_beaver_data.Q0[:, j:j + 1])

        # g0 = np.array(p0_beaver_data.M0[k*B:k*B+B]).transpose().dot(f_star)+\
        # np.array(E[k*B:k*B+B]).transpose().dot(p0_beaver_data.P0[:,j:j+1])+\
        # np.array(p0_beaver_data.Q0[:,j:j+1])

        # TODO: update model weights
        w0 = w0 - learning_rate / B * g0
        logging.info('FinishIteration{}'.format(j))
    # Reconstruct w
    send_to_P1(P0ReconstructW(w0))
    logging.info('FinishTraining')
    logging.debug('Final w0={}'.format(w0))
    print('SendRecontructWToP0')
    # write_to_file('P0.csv', np.concatenate(observe_lst,axis=1))


def p0_deal_data(data):
    # TODO write more comments for this part.
    global dimension_para
    global X0
    global E0
    global E
    global y0
    global w0
    global Xtest
    global Ytest
    if data.data_type == 'DistributeData':
        # Received data share from the data source.
        dimension_para = data.dim_para
        X0 = data.X
        y0 = data.y
        Xtest = data.Xtest
        Ytest = data.Ytest
        print('ReceivedDataFromDataSource n=%d,d=%d,t=%d, B=%d' \
              % (dimension_para.n, dimension_para.d, dimension_para.t, dimension_para.B))
        # Send data to P1 and Carlo to generate the Beaver triples.
        p0_local_generate(dimension_para)
        return True
    elif data.data_type == 'P1ToP0':
        print('ReceiveDataFromP1')
        q_data.put(data)
        s_data.release()
        return True
    elif data.data_type == 'CarloToP0':
        print('ReceiveDataFromCarlo')
        q_data.put(data)
        s_data.release()
        return True
    elif data.data_type == 'P1StartTraining':
        # Received starting training signal from P1
        s_start_train.acquire()
        print('ReceiveStartTrainingFromP1')
        # start training
        # reconstruct matrix E
        E0 = X0 - p0_beaver_data.M0
        send_to_P1(P0ReconstructE(E0))
        s_E0.release()
        logging.debug('E0IsAvailable')
        return True
    elif data.data_type == 'P1ReconstructE':
        print('ReceiveRecontructEFromP1')
        s_E0.acquire()
        logging.debug('E0.shape={}'.format(E0.shape))
        logging.debug('data.E1.shape={}'.format(data.E1.shape))
        E = E0 + data.E1
        # start training now
        s_train_loop.release()
        return True
    elif data.data_type == 'P1ReconstructFj':
        logging.info('ReceiveP1ReconstructFj')
        q_fj_train.put(data)
        s_fj_train.release()
        logging.debug('AllowToUseTheFjData')
        return True
    elif data.data_type == 'P1ReconstructFStarj':
        logging.info('ReceiveP1ReconstructFStarj')
        q_fstarj_train.put(data)
        s_fstarj_train.release()
        return True
    elif data.data_type == 'P1ReconstructW':
        print('ReceiveP1ReconstructW')
        w = data.w1 + w0
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


class P0DealThread(threading.Thread):
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
            if not p0_deal_data(data):
                break
        print('FinishP0DealThread')


def P0_deal_partial_beaver_data(data1, data2):
    # TODO consider this part again.
    global p0_beaver_data
    if data1.data_type == 'P1ToP0' and data2.data_type == 'CarloToP0':
        p1_to_p0 = data1
        c_to_p0 = data2
    elif data1.data_type == 'CarloToP0' and data2.data_type == 'P1ToP0':
        c_to_p0 = data1
        p1_to_p0 = data2
    else:
        print('IncorrectDataType')
        exit(-1)
    # do the multiplication now then send to P1 start signal
    n = dimension_para.n
    d = dimension_para.d
    t = dimension_para.t
    B = dimension_para.B
    O0 = np.zeros((B, t))
    Q0 = np.zeros((d, t))
    epoches = n // B
    M0, M00, M01 = p0_beaver_data.M0, p0_beaver_data.M00, p0_beaver_data.M01
    N0, N00, N01 = p0_beaver_data.N0, p0_beaver_data.N00, p0_beaver_data.N01
    P0, P00, P01 = p0_beaver_data.P0, p0_beaver_data.P00, p0_beaver_data.P01
    for j in range(t):
        k = j % epoches
        # what is the first and last two
        O0[:, j:j + 1] = np.array(M0[k * B:k * B + B]).dot(np.array(N0[:, j:j + 1])) + \
                         np.array(M00[k * B:k * B + B]).dot(np.array(p1_to_p0.N10[:, j:j + 1])) + \
                         np.array(M01[k * B: k * B + B]).dot(np.array(p1_to_p0.N10[:, j:j + 1])) + \
                         np.array(c_to_p0.O0[:, j:j + 1]) + \
                         np.array(p1_to_p0.M10[k * B: k * B + B]).dot(np.array(N00[:, j:j + 1])) + \
                         np.array(p1_to_p0.M10[k * B: k * B + B]).dot(np.array(N01[:, j:j + 1]))

        Q0[:, j:j + 1] = np.array(M0[k * B:k * B + B]).transpose().dot(np.array(P0[:, j:j + 1])) + \
                         np.array(M00[k * B:k * B + B]).transpose().dot(np.array(p1_to_p0.P10[:, j:j + 1])) + \
                         np.array(M01[k * B:k * B + B]).transpose().dot(np.array(p1_to_p0.P10[:, j:j + 1])) + \
                         np.array(c_to_p0.Q0[:, j:j + 1]) + \
                         np.array(p1_to_p0.M10[k * B:k * B + B]).transpose().dot(np.array(P00[:, j:j + 1])) + \
                         np.array(p1_to_p0.M10[k * B:k * B + B]).transpose().dot(np.array(P01[:, j:j + 1]))
    p0_beaver_data.O0 = O0
    p0_beaver_data.Q0 = Q0
    send_to_P1(P0StartTraining())
    print('FinishGenBeaverTriples&SendStartTrainingSignalToP1')
    s_start_train.release()
    print('M0.shape=%s,N0.shape=%s,O0.shape=%s,P0.shape=%s,Q0.shape=%s' \
          % (p0_beaver_data.M0.shape, p0_beaver_data.N0.shape, \
             p0_beaver_data.O0.shape, p0_beaver_data.P0.shape, p0_beaver_data.Q0.shape))

    logging.info('P0M0={}'.format(p0_beaver_data.M0))
    logging.info('P0N0={}'.format(p0_beaver_data.N0))
    logging.info('P0O0={}'.format(p0_beaver_data.O0))
    logging.info('P0P0={}'.format(p0_beaver_data.P0))
    logging.info('P0Q0={}'.format(p0_beaver_data.Q0))


class WaitBeaverDataThread(threading.Thread):
    # wait for the Beaver triple data thread
    def run(self):
        s_data.acquire()
        s_data.acquire()
        data1 = q_data.get(block=True)
        data2 = q_data.get(block=True)
        # Now deal with both data
        P0_deal_partial_beaver_data(data1, data2)


q = Queue()
s = threading.Semaphore(value=0)
# s_data and q_data are used to wait for the data from Carlo and P1
# WaitBeaverDataThread waits for s_data and does the remaining 
# computation to generate the Beaver triples.
s_data = threading.Semaphore(value=0)
q_data = Queue()
# s_start_train is used to wait for the completion of generation of Beaver triples.
s_start_train = threading.Semaphore(value=0)

s_train = threading.Semaphore(value=0)
q_train = Queue()
# s_E0 is used to wait for the starting the reconstructE operation.
s_E0 = threading.Semaphore(value=0)
# s_train_loop is used to wait for the completion of the reconstructE operation.
s_train_loop = threading.Semaphore(value=0)
# s_fj_train is used to wait for the reconstructFj operation.
s_fj_train = threading.Semaphore(value=0)
q_fj_train = Queue()
s_fstarj_train = threading.Semaphore(value=0)
q_fstarj_train = Queue()
dimension_para = None
X0 = None
y0 = None
E0 = None
E = None
w0 = None
# p1_data = None
Carlo_data = None
server = None
Xtest = None
Ytest = None
p0_beaver_data = None
observe_lst = []
if __name__ == '__main__':
    init_logging('logger_P0.log')
    print(cur_time_str())
    logging.info(cur_time_str())
    HOST, PORT = read_config().P0
    print('P0ServerInfo {}:{}'.format(HOST, PORT))
    server = ThreadedTCPServer((HOST, PORT), ThreadedTCPRequestHandler)
    # receive data thread
    server_thread = threading.Thread(target=server.serve_forever)
    # deal received data thread
    p0_deal_thread = P0DealThread()
    # training thread
    train_thread = threading.Thread(target=train)
    # wait for the initial data thread
    wait_beaver_data_thread = WaitBeaverDataThread()
    thread_lst = (server_thread, p0_deal_thread, train_thread, wait_beaver_data_thread)
    start_all_threads(thread_lst)
