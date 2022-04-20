import numpy as np

# batch_size = 4  # Length of document(num of sentence)
# timestep = 10  # Length of sentence(num of word)
# embedding_dim = 128  # Length of embedding_vector(one of word)

# batch_input_shape = np.random.random((batch_size, timestep, embedding_dim))
#
# kernel_size = [3, 3, 4]
# strides = 1

def CNN_1D(batch_input_shape: np.ndarray, kernel_size: list[int], strides: int):
    # default pooling: max pooling
    try:
        embedding_dim = batch_input_shape.shape[2]
    except IndexError:
        raise IndexError("batch_input_shape must have shape:(batch_size, timestep, embedding_dim)")

    result_vector = []
    for input_shape in batch_input_shape:
        # 각 문장의 배열을 strides만큼 패딩(0으로된 배열을 채워넣음)
        for i in range(strides):
            input_shape = np.insert(input_shape, 0, np.zeros(embedding_dim), axis=0)
            input_shape = np.append(input_shape, np.zeros(embedding_dim), axis=0)

        # 커널사이즈마다 컨볼루션 벡터를 생성 후 저장
        convolutionVectorArr = []
        for kernel in kernel_size:
            # 각 커널은 칸마다 가중치를 가지나, 이 코드에선 미구현됨.
            # input_shape를 돌며 각 위치마다 convolution작업 시행.
            kernel //= 2
            convolutionVector = []
            for pos, _ in enumerate(input_shape):
                kernel_valeu_vec = np.zeros(len(input_shape[0]))

                # 현 위치에서 커널사이즈만큼 주변 벡터를 더함.
                if pos - kernel < 0 or pos + kernel >= len(input_shape):
                    continue
                for value in input_shape[pos - kernel:pos + kernel]:
                    kernel_valeu_vec += value

                # 더해진 벡터를 평균해 convolutionVector에 추가
                kernel_value = sum(kernel_valeu_vec) / len(kernel_valeu_vec)
                convolutionVector.append(kernel_value)

            # 생성된 벡터를 전체 벡터에 더함
            convolutionVectorArr.append(convolutionVector)

        # 맥스풀링 > 연결(concatenate)
        for vector in convolutionVectorArr:
            result_vector.append(max(vector))
        # 이는 문장으로부터 얻은 최종 특성 벡터. 이를 출력층(벡터를 받아 훈련 후 다음층으로 이어지는 층)으로 연결.
        return result_vector
