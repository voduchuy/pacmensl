/*
MIT License

Copyright (c) 2020 Huy Vo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include "StateSet.h"

pacmensl::StateSet::StateSet( MPI_Comm comm, pacmensl::StateSetType type, int num_species,
                              pacmensl::PartitioningType part, pacmensl::PartitioningApproach repart ) {
    type_ = type;
    switch ( type ) {
        case BASE:
            state_set_ = new StateSetBase(comm);
            break;
        case CONSTRAINED:
            state_set_ = new StateSetConstrained(comm);
            break;
    }
}

void pacmensl::StateSet::Expand( ) {
    state_set_->Expand();
}

pacmensl::StateSet::~StateSet( ) {
    switch ( type_ ) {
        case BASE:
            delete state_set_;
            break;
        case CONSTRAINED:
            delete ((StateSetConstrained*) state_set_);
            break;
    }
}
