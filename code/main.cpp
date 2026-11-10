#include <iostream>
#include <vector>
#include <string>

// Configurações da CImg (sempre ANTES do #include "CImg.h")
#define cimg_display 0 
#define cimg_use_jpeg
#include "CImg.h"

// Inclusão da OpenCV
#include <opencv2/opencv.hpp>


using namespace cimg_library;

int main(int argc, char** argv) {
    // Caminho da imagem de entrada
    const char* filename = (argc > 1) ? argv[1] : "elementos.jpg";

    try {
        // =========================================================================
        // PASSO 1: Pré-processamento com CImg (Escala de Cinza + Filtro Gaussiano)
        // =========================================================================
        std::cout << "[Passo 1] Carregando imagem e aplicando pre-processamento via CImg..." << std::endl;
        
        CImg<unsigned char> img_original(filename);
        int largura = img_original.width();
        int altura = img_original.height();

        // Cria imagem de 1 canal (tons de cinza)
        CImg<unsigned char> img_gray(largura, altura, 1, 1, 0);

        // Conversão manual/explícita pela fórmula de luminância (ITU-R BT.601)
        cimg_forXY(img_original, x, y) {
            unsigned char r = img_original(x, y, 0, 0);
            unsigned char g = img_original(x, y, 0, 1);
            unsigned char b = img_original(x, y, 0, 2);
            img_gray(x, y) = static_cast<unsigned char>(0.299f * r + 0.587f * g + 0.114f * b);
        }
        img_gray.save("etapa1_a_cinza.jpg");

        // Fator de escala relativo à resolução base de referência (~1280x720)
        // Permite que o código funcione com a mesma precisão tanto em imagens HD quanto em fotos 4K
        double fator_escala = std::max(1.0, std::max(largura, altura) / 1280.0);
        double fator_area = (static_cast<double>(largura) * altura) / (1280.0 * 720.0);

        // Aplicação do Filtro Gaussiano via CImg
        // Sigma = 2.0 é ideal para suavizar a textura áspera da bancada metálica
        // sem destruir as bordas estruturais dos parafusos e porcas.
        CImg<unsigned char> img_blur = img_gray.get_blur(2.0f);
        img_blur.save("etapa1_b_gaussiano.jpg");


        // =========================================================================
        // PONTE CImg -> OpenCV
        // =========================================================================
        // Como img_blur tem apenas 1 canal (grayscale), o layout de memória planar 
        // da CImg é idêntico ao layout intercalado da OpenCV (cv::Mat CV_8UC1).
        // Usamos IMREAD_IGNORE_ORIENTATION para evitar que a tag EXIF do celular
        // rotacione a imagem na OpenCV de forma divergente da CImg.
        cv::Mat mat_original = cv::imread(filename, cv::IMREAD_COLOR | cv::IMREAD_IGNORE_ORIENTATION);
        cv::Mat mat_blur(altura, largura, CV_8UC1, img_blur.data());


        // =========================================================================
        // PASSO 2: Segmentação e Detecção de Bordas
        // =========================================================================
        std::cout << "[Passo 2] Executando segmentacao e deteccao de bordas..." << std::endl;

        // 2.1 Segmentação: Limiarização Adaptativa (isola variações locais em relação ao fundo)
        // Usamos THRESH_BINARY_INV para que as bordas/peças escuras e sombras 3D fiquem brancas (255) e o fundo preto (0)
        int block_size = static_cast<int>(51 * fator_escala);
        if (block_size % 2 == 0) block_size++; // O tamanho do bloco deve ser ímpar
        cv::Mat mat_segmentada;
        cv::adaptiveThreshold(mat_blur, mat_segmentada, 255, 
                              cv::ADAPTIVE_THRESH_GAUSSIAN_C, 
                              cv::THRESH_BINARY_INV, block_size, 16);
        cv::imwrite("etapa2_a_segmentacao_adaptativa.jpg", mat_segmentada);

        // 2.2 Detector de Bordas de Canny
        // O Canny captura tanto peças mais claras quanto mais escuras que o fundo
        cv::Mat mat_bordas;
        double limiar_inferior = 35.0;
        double limiar_superior = 100.0;
        cv::Canny(mat_blur, mat_bordas, limiar_inferior, limiar_superior);
        cv::imwrite("etapa2_b_bordas_canny.jpg", mat_bordas);

        // 2.3 Pós-processamento Morfológico (Essencial para unir as roscas dos parafusos)
        // As espirais da rosca geram dezenas de pequenas bordas quebradas. 
        // O fechamento morfológico (Dilatação seguida de Erosão) consolida cada peça em um sólido único.
        cv::Mat mat_morfologia;
        cv::Mat kernel_dilatacao = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
        cv::dilate(mat_bordas, mat_morfologia, kernel_dilatacao);

        // Combina as bordas nítidas do Canny com as estruturas fortes da segmentação adaptativa
        // (após abertura para eliminar poros da bancada), garantindo a detecção de peças escuras/oxidadas
        cv::Mat mat_seg_limpa;
        int tam_abertura_seg = std::max(3, static_cast<int>(std::round(3 * fator_escala)));
        if (tam_abertura_seg % 2 == 0) tam_abertura_seg++;
        cv::morphologyEx(mat_segmentada, mat_seg_limpa, cv::MORPH_OPEN,
                         cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(tam_abertura_seg, tam_abertura_seg)));
        cv::bitwise_or(mat_morfologia, mat_seg_limpa, mat_morfologia);

        // Remove pequenos fragmentos isolados de ruído/reflexo da bancada antes do fechamento
        cv::Mat labels, stats, centroids;
        int num_labels = cv::connectedComponentsWithStats(mat_morfologia, labels, stats, centroids, 8, CV_32S);
        int min_pixels_borda = static_cast<int>(45 * fator_area);
        for (int lbl = 1; lbl < num_labels; lbl++) {
            if (stats.at<int>(lbl, cv::CC_STAT_AREA) < min_pixels_borda) {
                mat_morfologia.setTo(0, labels == lbl);
            }
        }

        int tam_fechamento = std::max(7, static_cast<int>(std::round(7 * fator_escala)));
        if (tam_fechamento % 2 == 0) tam_fechamento++;
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(tam_fechamento, tam_fechamento));
        cv::morphologyEx(mat_morfologia, mat_morfologia, cv::MORPH_CLOSE, kernel, cv::Point(-1, -1), 2);

        // Preenche o interior dos contornos fechados para formar sólidos reais antes da abertura
        std::vector<std::vector<cv::Point>> contornos_temp;
        cv::findContours(mat_morfologia, contornos_temp, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        for (size_t i = 0; i < contornos_temp.size(); i++) {
            if (cv::contourArea(contornos_temp[i]) > 300.0 * fator_area) {
                cv::drawContours(mat_morfologia, contornos_temp, static_cast<int>(i), cv::Scalar(255), cv::FILLED);
            }
        }
        
        // Preenchimento e limpeza de pequenos ruídos da mesa com abertura morfológica
        int tam_abertura = std::max(3, static_cast<int>(std::round(4 * fator_escala)));
        if (tam_abertura % 2 == 0) tam_abertura++;
        cv::Mat kernel_limpeza = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(tam_abertura, tam_abertura));
        cv::morphologyEx(mat_morfologia, mat_morfologia, cv::MORPH_OPEN, kernel_limpeza);
        cv::imwrite("etapa2_c_morfologia_consolidada.jpg", mat_morfologia);


        // =========================================================================
        // PASSO 3: Detecção por Aproximações Poligonais e Contagem (OpenCV)
        // =========================================================================
        std::cout << "[Passo 3] Extraindo contornos, aproximando poligonos e contando..." << std::endl;

        std::vector<std::vector<cv::Point>> contornos;
        std::vector<cv::Vec4i> hierarquia;

        // RETR_EXTERNAL busca apenas o contorno externo de cada peça (ignora o furo interno das porcas/arruelas
        // para não contar a mesma peça duas vezes: uma pelo lado de fora e outra pelo furo de dentro).
        cv::findContours(mat_morfologia, contornos, hierarquia, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        cv::Mat mat_resultado = mat_original.clone();
        int contador_pecas = 0;

        // Área mínima em pixels proporcional à resolução da imagem para filtrar sujeiras ou riscos na bancada
        // (Para ~720x1280, ~600 px² é um ótimo corte; em 4K escala proporcionalmente)
        const double AREA_MINIMA = 650.0 * std::max(1.0, fator_area); 
        int espessura_linha = std::max(2, static_cast<int>(std::round(2.0 * fator_escala)));
        int raio_vertice = std::max(3, static_cast<int>(std::round(3.5 * fator_escala)));
        double escala_fonte = 0.55 * fator_escala;

        for (size_t i = 0; i < contornos.size(); i++) {
            double area = cv::contourArea(contornos[i]);
            
            if (area < AREA_MINIMA) {
                continue; // Ignora ruídos da bancada
            }

            contador_pecas++;

            // Cálculo do perímetro do contorno original
            double perimetro = cv::arcLength(contornos[i], true);

            // Aproximação Poligonal (Algoritmo de Ramer-Douglas-Peucker)
            // O fator 0.015 (1.5% do perímetro) define a precisão da aproximação
            std::vector<cv::Point> poligono_aproximado;
            cv::approxPolyDP(contornos[i], poligono_aproximado, 0.015 * perimetro, true);

            // 1. Desenha o contorno real suave em Verde
            cv::drawContours(mat_resultado, contornos, static_cast<int>(i), cv::Scalar(0, 255, 0), espessura_linha);

            // 2. Desenha a aproximação poligonal em Vermelho/Laranja para visualização técnica
            std::vector<std::vector<cv::Point>> poly_vec = { poligono_aproximado };
            cv::drawContours(mat_resultado, poly_vec, 0, cv::Scalar(0, 140, 255), espessura_linha);

            // 3. Destaca os vértices do polígono aproximado em Azul
            for (const cv::Point& p : poligono_aproximado) {
                cv::circle(mat_resultado, p, raio_vertice, cv::Scalar(255, 0, 0), -1);
            }

            // 4. Calcula o centro de massa (Momentos de Imagem) para posicionar o rótulo da contagem
            cv::Moments m = cv::moments(contornos[i]);
            if (m.m00 != 0) {
                int cx = static_cast<int>(m.m10 / m.m00);
                int cy = static_cast<int>(m.m01 / m.m00);

                std::string label = "#" + std::to_string(contador_pecas) + 
                                    " (V:" + std::to_string(poligono_aproximado.size()) + ")";
                
                int offset_x = static_cast<int>(35 * fator_escala);
                cv::putText(mat_resultado, label, cv::Point(cx - offset_x, cy), 
                            cv::FONT_HERSHEY_SIMPLEX, escala_fonte, cv::Scalar(255, 255, 255), espessura_linha + 2);
                cv::putText(mat_resultado, label, cv::Point(cx - offset_x, cy), 
                            cv::FONT_HERSHEY_SIMPLEX, escala_fonte, cv::Scalar(0, 0, 0), espessura_linha);
            }
        }

        // Escreve o placar final no topo da imagem
        std::string texto_total = "Total de elementos identificados: " + std::to_string(contador_pecas);
        int box_w = static_cast<int>(520 * fator_escala);
        int box_h = static_cast<int>(60 * fator_escala);
        cv::rectangle(mat_resultado, cv::Point(10, 10), cv::Point(10 + box_w, 10 + box_h), cv::Scalar(0, 0, 0), -1);
        cv::putText(mat_resultado, texto_total, cv::Point(25, 10 + static_cast<int>(40 * fator_escala)), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.8 * fator_escala, cv::Scalar(0, 255, 0), espessura_linha);

        cv::imwrite("etapa3_resultado_final.jpg", mat_resultado);

        std::cout << "=========================================================" << std::endl;
        std::cout << "Processamento concluido com sucesso!" << std::endl;
        std::cout << texto_total << std::endl;
        std::cout << "Imagens geradas:" << std::endl;
        std::cout << "  1. etapa1_a_cinza.jpg" << std::endl;
        std::cout << "  2. etapa1_b_gaussiano.jpg" << std::endl;
        std::cout << "  3. etapa2_a_segmentacao_adaptativa.jpg" << std::endl;
        std::cout << "  4. etapa2_b_bordas_canny.jpg" << std::endl;
        std::cout << "  5. etapa2_c_morfologia_consolidada.jpg" << std::endl;
        std::cout << "  6. etapa3_resultado_final.jpg" << std::endl;
        std::cout << "=========================================================" << std::endl;

    } catch (CImgException& e) {
        std::cerr << "Erro na CImg: " << e.what() << std::endl;
        return -1;
    } catch (cv::Exception& e) {
        std::cerr << "Erro na OpenCV: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}