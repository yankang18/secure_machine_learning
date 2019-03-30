import socketserver

from legacy_code.common_data import *
from queue import Queue


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


def carlo_deal_data(data1, data2):
    if data1.data_type == 'P0ToCarlo' and data2.data_type == 'P1ToCarlo':
        p0_to_carlo = data1
        p1_to_carlo = data2
    elif data1.data_type == 'P1ToCarlo' and data2.data_type == 'P0ToCarlo':
        p0_to_carlo = data2
        p1_to_carlo = data1
    else:
        print('IncorrectDataType')
        exit(-1)
    dimension_para = p0_to_carlo.dim_para
    n = dimension_para.n
    d = dimension_para.d
    t = dimension_para.t
    B = dimension_para.B
    O = np.zeros((B, t))
    Q = np.zeros((d, t))

    M00, N00, P00 = p0_to_carlo.M00, p0_to_carlo.N00, p0_to_carlo.P00
    M11, N11, P11 = p1_to_carlo.M11, p1_to_carlo.N11, p1_to_carlo.P11
    epoches = n // B
    print('N11.shape={}'.format(N11.shape))
    print('N11.column={}'.format(np.array(N11[:, 0:1]).shape))
    # TODO: should we use beaver triple ?
    # TODO: why do this using cross over ?
    for j in range(t):
        k = j % epoches
        O[:, j:j + 1] = np.array(M00[k * B:k * B + B]).dot(np.array(N11[:, j:j + 1]))
        Q[:, j:j + 1] = np.array(M00[k * B:k * B + B]).transpose().dot(np.array(P11[:, j:j + 1]))

        O[:, j:j + 1] = O[:, j:j + 1] + np.array(M11[k * B:k * B + B]).dot(np.array(N00[:, j:j + 1]))
        Q[:, j:j + 1] = Q[:, j:j + 1] + np.array(M11[k * B:k * B + B]).transpose().dot(np.array(P00[:, j:j + 1]))

    # TODO using O and Q here is misleading. They are actually NOT O and Q
    # split O and Q into two shares for the two parties respectively.
    O0 = generate_random_matrix(B, t)
    Q0 = generate_random_matrix(d, t)
    O1 = O - O0
    Q1 = Q - Q0
    send_to_P0(CarloToP0(O0, Q0))
    print('SendToP0')
    send_to_P1(CarloToP1(O1, Q1))
    print('SendToP1')


class CarloDealThread(threading.Thread):
    """
    This thread deals the received data.
    """

    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        s.acquire()
        data1 = load_obj(q.get(block=True))
        print('ReceivedDataType={}'.format(data1.data_type))
        s.acquire()
        data2 = load_obj(q.get(block=True))
        print('ReceivedDataType={}'.format(data2.data_type))
        carlo_deal_data(data1, data2)
        print('FinishCarloDealThread')
        server.shutdown()
        print(cur_time_str())


q = Queue()
s = threading.Semaphore(value=0)
server = None
if __name__ == '__main__':
    print(cur_time_str())
    HOST, PORT = read_config().Carlo
    print('CarloServerInfo {}:{}'.format(HOST, PORT))
    server = ThreadedTCPServer((HOST, PORT), ThreadedTCPRequestHandler)
    # receive data thread
    server_thread = threading.Thread(target=server.serve_forever)
    # deal received data thread
    carlo_deal_thread = CarloDealThread()
    thread_lst = (server_thread, carlo_deal_thread)
    start_all_threads(thread_lst)
