#lang racket

(require "util.rkt")
(require plot)

(define (linear-map x a b alpha beta)
  (let* ((k (/ (- beta alpha) (- b a)))
         (offset (- alpha (* k a))))
    (+ (* k x) offset)))

(define (random-float)
  (/ (exact->inexact (random 1000000000))
     (exact->inexact 1000000000)))

(define random-real random-float)

;Простенький генератор случайных чисел распределённых по
;~нормальному закону, работает из-за центральной предельной теоремыы
(define (randn mean variance)
  (let ((sum 0.0))
    (loop i 0 11
          (incf sum (random-float)))
    (+ mean (* variance (- sum 6.0)))))

(define random-normal randn)

(define (2dv-rotate phi v)
  (let ((x (vector-ref v 0))
        (y (vector-ref v 1)))
    (vector (- (* x (cos phi)) (* y (sin phi)))
            (+ (* x (sin phi)) (* y (cos phi))))))

(define (make-normal center variance) (λ () (random-normal center variance)))

(define (make-uniform alpha beta)
  (λ () (linear-map (random-real) 0.0 1.0 alpha beta)))

(define (make-2dnormal x-var y-var phi)
  (lambda () (2dv-rotate phi
                         (vector (random-normal 0 x-var)
                                 (random-normal 0 y-var)))))

(define (make-2dcircular-distr radial-distr angle-distr)
  (lambda () (2dv-rotate (angle-distr)
                         (vector (radial-distr) 0.0))))

(define (create-point-cluster distribution-proc center-vec n)
  (build-list n (λ (i) (vector-map + center-vec (distribution-proc)))))

(define (vector-foldl fn vec)
  (let ((sum (vector-ref vec 0)))
    (loop i 1 (- (vector-length vec) 1)
          (set! sum (fn sum (vector-ref vec i))))
    sum))

;============================================================================================

(define (gen-training N)
  (let ((class+1 (create-point-cluster (make-2dnormal 1.0 2.0 (/ pi -4)) #(3.0 4.0) (/ N 2)))
        (class-1 (create-point-cluster (make-2dnormal 1.0 1.0 0) #(0.0 -1.0) (/ N 2))))
    (printf "Generated dataset:\n~s\n"
     (plot (mix
            (points class+1 #:sym 'fullsquare #:color 'red)
            (points class-1 #:sym 'fullsquare #:color 'blue))))
    (append (map (λ (x) (list x 1)) class+1)
            (map (λ (x) (list x -1)) class-1))))

(define (classifier-quality classifier test-set)
  (/ (exact->inexact (length (filter (λ (x) (= (second x) (classifier (first x)))) test-set)))
     (exact->inexact (length test-set))))

(define (train-simple-perceptron dim bias epochs learning-rate training-set)
  (let* ((W (vector-map (lambda (x) (* 2.0 (- (random-float) 0.5))) (make-vector dim)))
         (N (length training-set))
         (bias bias))
    (printf "Training single layer perceptron by error gradient\n")
    (printf "Initializing random weights ~s\n" W)
    (loop i 1 epochs
          (let ((errors 0))
            (loop j 0 (- N 1)
                  (let* ((example (list-ref training-set j))
                         (x (first example))
                         (y (second example))
                         (y-perc (sgn (+ bias (vector-foldl + (vector-map * W x)))))
                         (err (- y y-perc)))
                    (when (not (or (eq? err 0.0) (eq? err 0)))
                      (set! errors (+ errors 1)))
                    (loop k 0 (- dim 1)
                          (let ((dW (* learning-rate (vector-ref x k) err)))
                            (vector-set! W k (+ (vector-ref W k) dW));))))
                            (set! bias (+ bias (* learning-rate err)))))))
            (printf "Epoch ~s\n" i)
            (printf "Error: ~s\n" (exact->inexact (/ errors N)))))
    (printf "Training complete\n")
    (printf "Final bias ~s weights ~s\n" bias W)
    (printf "~s\n"
     (plot (mix
            (points (map car (filter (λ (x) (eq? (second x) 1)) training-set))
                    #:sym 'fullsquare #:color 'red)
            (points (map car (filter (λ (x) (eq? (second x) -1)) training-set))
                         #:sym 'fullsquare #:color 'blue)
            (function (λ (x) (/ (- (- bias) (* x (vector-ref W 0)))
                                (vector-ref W 1)))
                      #:color 8 #:width 3 #:label "Decision boundary"))))
    (λ (x)
      (sgn (+ bias (vector-foldl + (vector-map * W x)))))))

(define (random-train dim epochs training-set)
  (let* ((best-W (vector-map (lambda (x) (* 2.0 (- (random-float) 0.5))) (make-vector dim)))
         (best-bias 0.0)
         (N (length training-set))
         (min-E N))
    (printf "Training single layer perceptron by random search\n")
    (loop i 0 epochs
          (let ((errors 0)
                (try-W (vector-map (lambda (x) (randn 0.0 5.0)) (make-vector dim)))
                (try-bias (randn 0.0 3.0)))
            (loop j 0 (- N 1)
                  (let* ((example (list-ref training-set j))
                         (x (first example))
                         (y (second example))
                         (y-perc (sgn (+ try-bias (vector-foldl + (vector-map * try-W x)))))
                         (err (- y y-perc)))
                    (when (not (= err 0))
                      (set! errors (+ errors 1)))))
            (when (< errors min-E)
              (printf "Error: ~s, b=~s, W=~s\n" (exact->inexact (/ errors N)) try-bias try-W)
              (begin
                (set! min-E errors)
                (set! best-W try-W)
                (set! best-bias try-bias)))))
    (printf "Training complete\n")
    (printf "Final bias ~s weights ~s\n" best-bias best-W)
    (printf "~s\n"
     (plot (mix
            (points (map car (filter (λ (x) (eq? (second x) 1)) training-set))
                    #:sym 'fullsquare #:color 'red)
            (points (map car (filter (λ (x) (eq? (second x) -1)) training-set))
                         #:sym 'fullsquare #:color 'blue)
            (function (λ (x) (/ (- (- best-bias) (* x (vector-ref best-W 0)))
                                (vector-ref best-W 1)))
                      #:color 8 #:width 3 #:label "Decision boundary"))))
    (λ (x)
      (sgn (+ best-bias (vector-foldl + (vector-map * best-W x)))))))


(define tr-set (gen-training 100))

(define P-grad (train-simple-perceptron 2 0.0 6 0.03 tr-set))
(define P-rand (random-train 2 50 tr-set))

(printf "Checking classifier quality\ngradient-perceptron:~s random-perceptron:~s\n"
        (classifier-quality P-grad tr-set)
        (classifier-quality P-rand tr-set))