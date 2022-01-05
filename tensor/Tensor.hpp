#include <stdint.h>
#include <limits>
#include <cassert>
#include <array>

template<uint32_t DDims>
class Address;

template<typename DType, uint32_t DDims>
class Tensor
{
public: 
    Tensor(const std::array<uint32_t, DDims>& dims) :
        dims_(dims),
        data_(new DType[size()])
    {
        steps_[DDims-1] = 1;
        for(int i = dims_.size() - 2; i >= 0; --i)
        {
            steps_[i] = dims_[i+1] * steps_[i+1];
        }
    }

    ~Tensor() 
    {
        if(data_) delete[] data_;
    }

    DType& operator[](const Address<DDims>& addr) const 
    {
        uint32_t dataIndex = address2DataIndex(addr);
        return data_[dataIndex];
    }

    uint32_t size() const
    {
        uint64_t dataSize = 1;
        for(int i = 0, imax = dims_.size(); i < imax; ++i) 
        {
            dataSize *= dims_[i];
            assert(dataSize <= std::numeric_limits<uint32_t>::max());
        }
        
        return static_cast<uint32_t>(dataSize);
    }

private: 
    uint32_t address2DataIndex(const Address<DDims>& addr) const
    {
        uint32_t dataIndex = 0;
        for(int i = dims_.size() - 1; i >= 0; --i)
        {
            dataIndex += addr[i] * steps_[i];
        }

        return dataIndex;
    }

private:
    const std::array<uint32_t, DDims> dims_;
    std::array<uint32_t, DDims> steps_;
    DType* const data_;
};

