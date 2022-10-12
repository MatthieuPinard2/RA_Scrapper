#include <iostream>
#include <mkl.h>
#include <vector>
#include <algorithm>
#include <assert.h>
#include <memory>

using BTime = int;

namespace AMC2 {

    template <class T>
    class Matrix {
    private:
        size_t m_nRows, m_nCols;
        T* m_data;
    public:
        Matrix(const size_t nRows, const size_t nCols) : m_nRows(nRows), m_nCols(nCols) {
            m_data = new T[m_nRows * m_nCols];
        }
        ~Matrix() {
            delete[](m_data);
        }
        inline size_t getNbCols() const {
            return m_nCols;
        }
        inline T* getRow(const size_t i) {
            return m_data + (m_nCols * i);
        }
        inline const T* getConstRow(const size_t i) const {
            return m_data + (m_nCols * i);
        }
        inline T& getElement(const size_t i, const size_t j) {
            return *(m_data + (m_nCols * i) + j);
        }
        inline T const& getConstElement(const size_t i, const size_t j) const {
            return *(m_data + (m_nCols * i) + j);
        }
    };

    enum class ExerciseType { NoExercise, Callable, Putable, Autocallable };

    class AMC2_Exercise {
    protected:
        double m_exitValue;
        size_t m_exerciseIndex;
        BTime m_settlementDate;
        BTime m_observationDate;
    public:
        inline double getExitValue() const {
            return m_exitValue;
        }
        inline size_t getExerciseIndex() const {
            return m_exerciseIndex;
        }
        inline BTime getExerciseSettlementDate() const {
            return m_settlementDate;
        }
        inline BTime getExerciseObservationDate() const {
            return m_observationDate;
        }
        virtual inline void computeExercise(
            std::vector<double>& trajectories,
            Matrix<double> const& performances,
            std::vector<double> const& regressedGain) const = 0;
    };

    /* No Exercise */
    class AMC2_Exercise_NoExercise : protected AMC2_Exercise {
    public:
        inline void computeExercise(
            std::vector<double>& trajectories,
            Matrix<double> const& performances,
            std::vector<double> const& regressedGain) const
        {}
    };

    /* Callable Exercise */
    class AMC2_Exercise_Callable : protected AMC2_Exercise {
    public:
        inline void computeExercise(
            std::vector<double>& trajectories,
            Matrix<double> const& performances,
            std::vector<double> const& regressedGain) const
        {
            // If regressedGain >= 0 (i.e. continuationValue >= callLevel), trajectories = callLevel.  
            for (size_t i = 0; i < trajectories.size(); ++i) {
                if (regressedGain[i] >= 0) {
                    trajectories[i] = m_exitValue;
                }
            }
        }
    };

    /* Putable Exercise */
    class AMC2_Exercise_Putable : protected AMC2_Exercise {
    public:
        inline void computeExercise(
            std::vector<double>& trajectories,
            Matrix<double> const& performances,
            std::vector<double> const& regressedGain) const
        {
            // If regressedGain <= 0 (i.e. continuationValue <= putLevel), trajectories = putLevel.  
            for (size_t i = 0; i < trajectories.size(); ++i) {
                if (regressedGain[i] <= 0) {
                    trajectories[i] = m_exitValue;
                }
            }
        }
    };

    /* Smoothing the exercise decision for Autocallables. */
    enum class BarrierType { UpBarrier, DownBarrier };
    enum class UnderlyingType { WorstOf, BestOf, Basket, Mono };
    
    class AMC2_Smoothing_Parameters {
    protected:
        size_t m_nUnderlyings;
        BarrierType m_barrierType;
        UnderlyingType m_underlyingType;
        std::vector<double> m_spreadMin;     // Floors the minimum smoothing width.
        std::vector<double> m_deltaMax;      // Maximum Delta Cash expressed in underlying currency, at today. 
        std::vector<double> m_barrierLevel;  // Barrier *relative* level per udl.
        std::vector<double> m_FX;            // Udl/Payoff FX spot rate. 
        double m_notional;                   // Notional in Payoff currency.
        double m_gearing;                    // Allows to apply gearing on a specific barrier.
        bool m_disableSmoothingAtToday;      // If barrierDate == modelDate and this flag is true OR barrierDate < modelDate, we don't smooth.
        std::vector<double> m_adjustedDMax;
        inline double callSpread(const double x) const {
            const double y = std::max(0.0, std::min(x, 1.0));
            return (y <= 0.5) ? 2.0 * y * y : (4.0 - 2.0 * y) * y - 1.0;
        }
        inline double callSpreadUnsmoothed(const double x) const {
            return x >= 0.0 ? 1.0 : 0.0;
        }
    public:
        AMC2_Smoothing_Parameters() {
            m_adjustedDMax.resize(m_nUnderlyings);
        }
        inline size_t getUnderlyingsCount() const {
            return m_nUnderlyings;
        }
        inline virtual double getSmoothing(const double regressedGain, const double* performances) const = 0;
    };

    class AMC2_Smoothing_Parameters_Mono : protected AMC2_Smoothing_Parameters {
    public:
        AMC2_Smoothing_Parameters_Mono() {
            assert(m_underlyingType == UnderlyingType::Mono);
            m_adjustedDMax[0] = m_deltaMax[0] * m_barrierLevel[0] * m_FX[0];
        }
        inline double getSmoothing(const double regressedGain, const double* performances) const {
            if (m_disableSmoothingAtToday) {
                if (m_barrierType == BarrierType::UpBarrier) {
                    return callSpreadUnsmoothed(+performances[0] - m_barrierLevel[0]);
                }
                else /* Down Barrier */ {
                    return callSpreadUnsmoothed(-performances[0] + m_barrierLevel[0]);
                }
            }
            else {
                const double premiumGap = regressedGain * m_notional;
                const double barrierShift = (premiumGap >= 0.0) ? 1.0 : 0.0;
                double epsilon = std::abs(premiumGap / m_adjustedDMax[0]);
                epsilon = std::max(epsilon, m_spreadMin[0]) * m_gearing;
                if (m_barrierType == BarrierType::UpBarrier) {
                    return callSpread((+performances[0] - m_barrierLevel[0] + (barrierShift * epsilon)) / epsilon);
                }
                else /* Down Barrier */ {
                    return callSpread((-performances[0] + m_barrierLevel[0] + (barrierShift * epsilon)) / epsilon);
                }
            }
        }
    };

    class AMC2_Smoothing_Parameters_Basket : protected AMC2_Smoothing_Parameters_Mono {
    private:
        std::vector<double> m_basketWeights;
    public:
        AMC2_Smoothing_Parameters_Basket(std::vector<double> const& basketWeights) : m_basketWeights(basketWeights) {
            assert(m_underlyingType == UnderlyingType::Basket);
            for (size_t i = 0; i < m_nUnderlyings; ++i) {
                m_adjustedDMax[0] = std::min(m_deltaMax[i] * m_barrierLevel[0] * m_FX[i] / m_basketWeights[i], m_adjustedDMax[0]);
            }
        }
    };

    class AMC2_Smoothing_Parameters_Multi : protected AMC2_Smoothing_Parameters {
    private:
        // Temporary buffers.
        mutable std::vector<double> m_individualSmoothings;
        bool m_globalIsMinOfIndividuals;
    public:
        AMC2_Smoothing_Parameters_Multi() {
            assert(m_underlyingType == UnderlyingType::BestOf || m_underlyingType == UnderlyingType::WorstOf);
            m_individualSmoothings.resize(m_nUnderlyings);
            for (size_t i = 0; i < m_nUnderlyings; ++i) {
                m_adjustedDMax[i] = m_deltaMax[i] * m_barrierLevel[i] * m_FX[i];
            }
            m_globalIsMinOfIndividuals =
                (m_barrierType == BarrierType::UpBarrier && m_underlyingType == UnderlyingType::WorstOf) ||
                (m_barrierType == BarrierType::DownBarrier && m_underlyingType == UnderlyingType::BestOf);
        }
        inline double getSmoothing(const double regressedGain, const double* performances) const {
            if (m_disableSmoothingAtToday) {
                if (m_barrierType == BarrierType::UpBarrier) {
                    for (size_t i = 0; i < m_nUnderlyings; ++i) {
                        m_individualSmoothings[i] = callSpreadUnsmoothed(+performances[i] - m_barrierLevel[i]);
                    }
                }
                else /* Down Barrier */ {
                    for (size_t i = 0; i < m_nUnderlyings; ++i) {
                        m_individualSmoothings[i] = callSpreadUnsmoothed(-performances[i] + m_barrierLevel[i]);
                    }
                }
            }
            else {
                const double premiumGap = regressedGain * m_notional;
                const double barrierShift = (premiumGap >= 0.0) ? 1.0 : 0.0;
                if (m_barrierType == BarrierType::UpBarrier) {
                    for (size_t i = 0; i < m_nUnderlyings; ++i) {
                        double epsilon = std::abs(premiumGap / m_adjustedDMax[i]);
                        epsilon = std::max(epsilon, m_spreadMin[i]) * m_gearing;
                        m_individualSmoothings[i] = callSpread((+performances[i] - m_barrierLevel[i] + (barrierShift * epsilon)) / epsilon);
                    }
                }
                else /* Down Barrier */ {
                    for (size_t i = 0; i < m_nUnderlyings; ++i) {
                        double epsilon = std::abs(premiumGap / m_adjustedDMax[i]);
                        epsilon = std::max(epsilon, m_spreadMin[i]) * m_gearing;
                        m_individualSmoothings[i] = callSpread((-performances[i] + m_barrierLevel[i] + (barrierShift * epsilon)) / epsilon);
                    }
                }
            }
            // Compute the global smoothing indicator based on the individual ones.
            if (m_globalIsMinOfIndividuals) {
                double multiSmoothing = 1.0;
                for (size_t i = 0; i < m_nUnderlyings; ++i) {
                    multiSmoothing = std::min(multiSmoothing, m_individualSmoothings[i]);
                }
                return multiSmoothing;
            }
            else { 
                double multiSmoothing = 0.0;
                for (size_t i = 0; i < m_nUnderlyings; ++i) {
                    multiSmoothing = std::max(multiSmoothing, m_individualSmoothings[i]);
                }
                return multiSmoothing;
            }
        }
    };

    /* Autocallable Exercise */
    class AMC2_Exercise_Autocallable : protected AMC2_Exercise {
        std::shared_ptr<AMC2_Smoothing_Parameters> m_smoothingParams;
    public:
        inline void computeExercise(
            std::vector<double>& trajectories,
            Matrix<double> const& performances,
            std::vector<double> const& regressedGain) const
        {
            for (size_t i = 0; i < trajectories.size(); ++i) {
                const double smoothing = m_smoothingParams->getSmoothing(regressedGain[i], performances.getConstRow(i));
                trajectories[i] = (1.0 - smoothing) * trajectories[i] + smoothing * m_exitValue;
            }
        }
    };

    class AMC2_Flow {
    private:
        size_t m_exerciseIndex;
        BTime m_settlementDate;
        BTime m_observationDate;
        double m_amount;
        double m_discountFactor; /* = DF(modelDate, Settlement of Flow) */
    public:
        inline size_t getExerciseIndex() const {
            return m_exerciseIndex;
        }
        inline BTime getFlowObservationDate() const {
            return m_observationDate;
        }
        inline BTime getFlowSettlementDate() const {
            return m_settlementDate;
        }
        inline double getDiscountedAmount() const {
            return m_amount * m_discountFactor;
        }
        AMC2_Flow(const size_t exerciseIndex,
                  const BTime settlementDate) : m_exerciseIndex(exerciseIndex), m_settlementDate(settlementDate) {}
        inline void setFlow(const double amount, const double DF) {
            m_amount = amount;
            m_discountFactor = DF;
        }
    };

    class AMC2_Engine {
    private:
        /* These ones should be set by the model... (per trajectory data) */
        std::vector<double> m_discountFactors; /* = DF(modelDate, Settlement of Exercise) */
        Matrix<double> m_performances;         /* = {S1_t/S1_0 ... SN_t/SN_0} for a WOF/BOF, Bsk_t/Bsk_0 for a Basket, S_t/S_0 for a Mono */
        Matrix<double> m_stateVariables;
        Matrix<double> m_linearStateVariables;
        Matrix<AMC2_Flow> m_flows;
        /* Contract data */
        std::vector<AMC2_Exercise> m_exercises;
        BTime m_modelDate;
        size_t m_polynomialDegree;
        bool m_includeFlowInRebate;
        /* These ones can be deduced from the former... */
        std::vector<double> m_trajectories;
        std::vector<double> m_regressedGain;
        size_t m_nFlows;
        size_t m_nExerciseDates;
        size_t m_nTrajectories;
        size_t m_basisSize; // = 1 + m_polynomialDegree * len(m_stateVariables) + len(m_linearStateVariables)
        Matrix<double> m_basis;
    public:
        inline void rescaleStateVariable(Matrix<double>& svMatrix) {
            const size_t nStateVariables = svMatrix.getNbCols();
            std::vector<double> x1(nStateVariables), x2(nStateVariables);
            for (size_t i = 0; i < m_nTrajectories; ++i) {
                for (size_t j = 0; j < nStateVariables; ++j) {
                    const double SV = svMatrix.getConstElement(i, j);
                    x1[j] += SV; x2[j] += SV * SV;
                }
            }
            for (size_t j = 0; j < nStateVariables; ++j) {
                x1[j] /= m_nTrajectories;
                x2[j] /= m_nTrajectories;
                x2[j] -= (x1[j] * x1[j]);
                x2[j] *= (m_nTrajectories / (m_nTrajectories - 1.0));
            }
            for (size_t i = 0; i < m_nTrajectories; ++i) {
                for (size_t j = 0; j < nStateVariables; ++j) {
                    svMatrix.getElement(i, j) -= x1[j];
                    svMatrix.getElement(i, j) /= sqrt(x2[j]);
                }
            }
        }
        inline void computeBasis() {
            for (size_t i = 0; i < m_nTrajectories; ++i) {
                size_t basisIdx = 0;
                m_basis.getElement(i, basisIdx++) = 1;
                for (size_t j = 0; j < m_stateVariables.getNbCols(); ++j) {
                    double SV = m_stateVariables.getConstElement(i, j); 
                    for (size_t k = 0; k < m_polynomialDegree; ++k) {
                        m_basis.getElement(i, basisIdx++) = SV;
                        SV *= SV;
                    }
                }
                for (size_t j = 0; j < m_linearStateVariables.getNbCols(); ++j) {
                    const double SV = m_linearStateVariables.getConstElement(i, j);
                    m_basis.getElement(i, basisIdx++) = SV;
                }
            }
        }
        inline double getTVBackward() {
            // First, populate m_flows with the flows that have m_settlementDate >= m_modelDate.
            // Exercise Index of a flow = i <=> Exercise Obs Date (i-1) < Flow Obs Date <= Exercise Obs Date (i)
            // (if flows are included in rebate, reverse the </<= otherwise)
            // Flows with Exercise Index == (m_nExerciseDates + 1) are bullet (paid regardless of an exercise event)
            size_t currentExIdx = m_nExerciseDates - 1;
            memset(m_trajectories.data(), 0, m_nTrajectories * sizeof(double));
            while (currentExIdx < m_nExerciseDates && m_exercises[currentExIdx].getExerciseObservationDate() >= m_modelDate) {
                // First, update m_discountFactors m_performances m_stateVariables m_stateVariables
                // (...)
                // Second, update m_trajectories by adding the flows in the current exercise period.
                // Remember that flows are discounted up to m_modelDate.
                // We rescale the discount up to the current exercise settlement date.
                // (...)
                for (size_t i = 0; i < m_nTrajectories; ++i) {
                    for (size_t j = 0; j < m_nFlows; ++j) {
                        AMC2_Flow const& thisFlow = m_flows.getConstElement(i, j);
                        if (thisFlow.getExerciseIndex() == currentExIdx + 1) {
                            m_trajectories[i] += thisFlow.getDiscountedAmount();
                        }
                    }
                    m_trajectories[i] /= m_discountFactors[i];
                }
                /* Rescale state variables */
                rescaleStateVariable(m_stateVariables);
                rescaleStateVariable(m_linearStateVariables);
                /* Basis computation */
                computeBasis();
                /* Compute regressed gain here. */

                /* Compute the exercise decision */
                m_exercises[currentExIdx].computeExercise(m_trajectories, m_performances, m_regressedGain);
                /* Discount m_trajectories up to m_modelDate. */
                for (size_t i = 0; i < m_nTrajectories; ++i) {
                    m_trajectories[i] *= m_discountFactors[i];
                }
                currentExIdx--;
            }
            // We add the remaining flows, as well as the bullet flows into the TV.
            currentExIdx++;
            for (size_t i = 0; i < m_nTrajectories; ++i) {
                for (size_t j = 0; j < m_nFlows; ++j) {
                    AMC2_Flow const& thisFlow = m_flows.getConstElement(i, j);
                    if (thisFlow.getExerciseIndex() == m_nExerciseDates + 1 || thisFlow.getExerciseIndex() <= currentExIdx) {
                        m_trajectories[i] += thisFlow.getDiscountedAmount();
                    }
                }
            }
            // The premium is the average over all trajectories.
            double TV = 0.0;
            for (size_t i = 0; i < m_nTrajectories; ++i) {
                TV += m_trajectories[i];
            }
            TV /= m_nTrajectories;
            return TV;
        }
    };
}

int main()
{
    std::cout << "Hello World!\n";
}
